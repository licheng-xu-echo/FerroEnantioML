#!/usr/bin/env python3
"""Run batchable nested validation tasks for descriptor/model screening.

Each task is one descriptor x model nested-validation run. A task internally
runs all 55 outer LOO folds for that descriptor/model pair, which makes the
screening easy to distribute across machines:

    /usr/bin/python3 nested_screen.py --screen ferr_lig --start 1 --end 10
    /usr/bin/python3 nested_screen.py --screen ferr_lig --start 11 --end 20

Use aggregate_nested_screen.py after all task ranges have finished.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import optuna
from lightgbm import LGBMRegressor
from optuna.samplers import TPESampler
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from rxnpredict.evaluate.eval import get_predict, get_val_score_add_data

warnings.filterwarnings("ignore")


MODEL_ORDER = ["gb", "rf", "et", "mlp", "ada", "bg", "dt", "xgb", "lgbm"]

OTHER_DESC_ORDER = ["MorganFingerprints", "RDKit", "SPOC", "Mordred"]

# Mirrors scripts/read_screen_results.ipynb: TS comes from final_descriptor_map
# (TS + SPOC for other components), while the other entries come from
# final_descriptor_ferrocene_ligand.
FERR_LIG_DESC_ORDER = [
    "TS",
    "ACSF",
    "MBTR",
    "Mordred",
    "MorganFingerprints",
    "RDKit",
    "SOAP",
    "SPOC",
]


@dataclass(frozen=True)
class Task:
    task_id: int
    screen: str
    descriptor: str
    model: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run nested validation descriptor/model screening tasks."
    )
    parser.add_argument(
        "--screen",
        choices=["other", "ferr_lig"],
        default="ferr_lig",
        help="Screening grid. 'other' reproduces 2D component descriptor screening; "
        "'ferr_lig' reproduces the final descriptor benchmark including TS.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="First 1-based task id to run, inclusive.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Last 1-based task id to run, inclusive.",
    )
    parser.add_argument(
        "--task-ids",
        type=str,
        default=None,
        help="Comma-separated 1-based task ids. Overrides --start/--end.",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print task count and a short manifest without running anything.",
    )
    parser.add_argument("--n-trials", type=int, default=15)
    parser.add_argument("--n-startup-trials", type=int, default=5)
    parser.add_argument(
        "--topk-values",
        type=str,
        default="100,200,300,400,446",
        help="Comma-separated top-k values optimized in the inner loop.",
    )
    parser.add_argument(
        "--inner-folds",
        type=int,
        default=5,
        help="Inner CV folds used for hyperparameter/top-k optimization. "
        "Default: 5-fold CV. Use 10 for 10-fold CV or 0 for inner LOO.",
    )
    parser.add_argument(
        "--objective-metric",
        choices=["r2", "mae"],
        default="r2",
        help="Metric optimized in inner validation. MAE is minimized.",
    )
    parser.add_argument(
        "--outer-folds",
        type=str,
        default=None,
        help="Optional comma-separated outer folds for smoke tests. "
        "Omit this in production to run all 55 outer folds.",
    )
    parser.add_argument("--merge-method", default="delta", choices=["delta", "mix", "dest"])
    parser.add_argument("--dist-type", default="euclidean")
    parser.add_argument("--random-state", type=int, default=1024)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/nested"),
        help="Output directory relative to the current working directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run tasks even if their output JSON already exists.",
    )
    parser.add_argument(
        "--optuna-verbosity",
        choices=["warning", "info", "debug"],
        default="warning",
    )
    return parser.parse_args()


def load_descriptor_data(screen: str, descriptor: str) -> dict[str, np.ndarray]:
    if screen == "other":
        data = np.load("../desc/final_descriptor_map.npy", allow_pickle=True).item()
        return data[descriptor]

    if descriptor == "TS":
        data = np.load("../desc/final_descriptor_map.npy", allow_pickle=True).item()
        return data["SPOC"]

    data = np.load("../desc/final_descriptor_ferrocene_ligand.npy", allow_pickle=True).item()
    return data[descriptor]


def build_tasks(screen: str) -> list[Task]:
    desc_order = OTHER_DESC_ORDER if screen == "other" else FERR_LIG_DESC_ORDER
    tasks: list[Task] = []
    task_id = 1
    for descriptor in desc_order:
        for model in MODEL_ORDER:
            tasks.append(
                Task(
                    task_id=task_id,
                    screen=screen,
                    descriptor=descriptor,
                    model=model,
                )
            )
            task_id += 1
    return tasks


def inner_cv_label(inner_folds: int) -> str:
    if inner_folds and inner_folds > 1:
        return f"inner{inner_folds}fold"
    return "innerloo"


def task_output_path(
    results_dir: Path, task: Task, inner_folds: int
) -> Path:
    safe_desc = task.descriptor.replace("/", "_")
    filename = (
        f"task_{task.task_id:03d}_{safe_desc}_{task.model}_"
        f"{inner_cv_label(inner_folds)}.json"
    )
    return results_dir / task.screen / "tasks" / filename


def suggest_model_params(trial: optuna.Trial, model_name: str) -> dict:
    if model_name in {"rf", "et"}:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 6),
            "max_depth": trial.suggest_int("max_depth", 3, 60),
        }
    if model_name == "mlp":
        return {
            "hidden_layer_sizes": trial.suggest_categorical(
                "hidden_layer_sizes", [(50,), (100,), (50, 100, 50), (100, 50, 100)]
            ),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "learning_rate_init": trial.suggest_categorical(
                "learning_rate_init", [0.0001, 0.001, 0.01]
            ),
        }
    if model_name == "ada":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_categorical(
                "learning_rate", [0.01, 0.1, 1.0]
            ),
        }
    if model_name == "bg":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 10, 100),
            "max_samples": trial.suggest_categorical("max_samples", [0.1, 0.5, 1.0]),
            "max_features": trial.suggest_categorical(
                "max_features", [0.1, 0.5, 1.0]
            ),
        }
    if model_name == "dt":
        return {
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_depth": trial.suggest_int("max_depth", 3, 60),
        }
    if model_name == "gb":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 6),
            "max_depth": trial.suggest_int("max_depth", 3, 60),
        }
    if model_name == "xgb":
        # Mirrors the current project search space.
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 6),
            "max_depth": trial.suggest_int("max_depth", 3, 60),
        }
    if model_name == "lgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "num_leaves": trial.suggest_int("num_leaves", 31, 60),
        }
    raise ValueError(f"Unknown model: {model_name}")


def make_model(model_name: str, params: dict, random_state: int):
    if model_name == "rf":
        return RandomForestRegressor(**params, random_state=random_state, n_jobs=-1)
    if model_name == "et":
        return ExtraTreesRegressor(**params, random_state=random_state, n_jobs=-1)
    if model_name == "mlp":
        return MLPRegressor(**params, random_state=random_state)
    if model_name == "ada":
        return AdaBoostRegressor(**params, random_state=random_state)
    if model_name == "bg":
        return BaggingRegressor(**params, random_state=random_state, n_jobs=-1)
    if model_name == "dt":
        return DecisionTreeRegressor(**params, random_state=random_state)
    if model_name == "gb":
        return GradientBoostingRegressor(**params, random_state=random_state)
    if model_name == "xgb":
        return XGBRegressor(**params, random_state=random_state, n_jobs=-1)
    if model_name == "lgbm":
        return LGBMRegressor(**params, random_state=random_state, n_jobs=-1, verbosity=-1)
    raise ValueError(f"Unknown model: {model_name}")


def inner_selection_info(args: argparse.Namespace) -> dict:
    if args.inner_folds and args.inner_folds > 1:
        return {
            "type": "cv",
            "fold": args.inner_folds,
            "random_state": args.random_state,
            "metric": ["r2", "mae"],
        }
    return {"type": "loo", "fold": 10, "metric": ["r2", "mae"]}


def run_outer_fold(
    task: Task,
    args: argparse.Namespace,
    base_X: np.ndarray,
    base_y: np.ndarray,
    target_X: np.ndarray,
    target_y: np.ndarray,
    outer_fold: int,
    topk_values: list[int],
    selection_inf: dict,
) -> dict:
    outer_mask = np.ones(target_X.shape[0], dtype=bool)
    outer_mask[outer_fold] = False
    train_X = target_X[outer_mask]
    train_y = target_y[outer_mask]
    test_X = target_X[[outer_fold]]
    test_y = target_y[[outer_fold]]

    def objective(trial: optuna.Trial) -> float:
        params = suggest_model_params(trial, task.model)
        topk = trial.suggest_categorical("topk", topk_values)
        model = make_model(task.model, params, args.random_state)
        _, _, score_map = get_val_score_add_data(
            model,
            base_X,
            base_y,
            train_X,
            train_y,
            selection_inf=selection_inf,
            merge_method=args.merge_method,
            topk=topk,
            dist_type=args.dist_type,
            verbose=False,
        )
        if args.objective_metric == "mae":
            return -float(score_map["mae"])
        return float(score_map["r2"])

    sampler = TPESampler(n_startup_trials=args.n_startup_trials, seed=args.random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    best_params = dict(study.best_params)
    best_topk = int(best_params.pop("topk"))
    best_model = make_model(task.model, best_params, args.random_state)
    pred = get_predict(
        best_model,
        base_X,
        base_y,
        train_X,
        train_y,
        test_X,
        merge_method=args.merge_method,
        simi_eval=True,
        dist_type=args.dist_type,
        topk=best_topk,
        verbose=False,
    )

    return {
        "outer_fold": int(outer_fold),
        "outer_train_shape": list(train_X.shape),
        "outer_test_shape": list(test_X.shape),
        "best": {
            "objective_value": float(study.best_value),
            "topk": best_topk,
            "model_params": best_params,
            "trial_number": int(study.best_trial.number),
        },
        "prediction": {
            "target_index": int(outer_fold),
            "y_true": float(test_y[0]),
            "y_pred": float(pred[0]),
            "absolute_error": float(abs(test_y[0] - pred[0])),
        },
    }


def run_task(task: Task, args: argparse.Namespace) -> dict:
    desc_data = load_descriptor_data(task.screen, task.descriptor)
    base_X = desc_data["base_X"]
    base_y = desc_data["base_y"]
    target_X = desc_data["target_X"]
    target_y = desc_data["target_y"]

    topk_values = [int(x) for x in args.topk_values.split(",") if x.strip()]
    selection_inf = inner_selection_info(args)

    folds = []
    outer_folds = selected_outer_folds(args, target_X.shape[0])
    for outer_fold in outer_folds:
        fold_start = time.time()
        print(
            f"[FOLD] task {task.task_id} {task.descriptor} {task.model} "
            f"outer={outer_fold}"
        )
        fold_payload = run_outer_fold(
            task,
            args,
            base_X,
            base_y,
            target_X,
            target_y,
            outer_fold,
            topk_values,
            selection_inf,
        )
        fold_payload["runtime_sec"] = round(time.time() - fold_start, 3)
        folds.append(fold_payload)

    return {
        "task": asdict(task),
        "settings": {
            "n_trials": args.n_trials,
            "n_startup_trials": args.n_startup_trials,
            "topk_values": topk_values,
            "inner_selection": selection_inf,
            "objective_metric": args.objective_metric,
            "merge_method": args.merge_method,
            "dist_type": args.dist_type,
            "random_state": args.random_state,
            "feature_selection": "none",
        },
        "data_shape": {
            "base_X": list(base_X.shape),
            "target_X": list(target_X.shape),
        },
        "outer_folds": outer_folds,
        "folds": folds,
    }


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as tmp:
        json.dump(payload, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def selected_task_ids(args: argparse.Namespace, total: int) -> list[int]:
    if args.task_ids:
        ids = [int(x) for x in args.task_ids.split(",") if x.strip()]
    else:
        start = args.start if args.start is not None else 1
        end = args.end if args.end is not None else total
        ids = list(range(start, end + 1))
    bad = [x for x in ids if x < 1 or x > total]
    if bad:
        raise ValueError(f"Task ids out of range 1..{total}: {bad[:10]}")
    return ids


def selected_outer_folds(args: argparse.Namespace, total: int) -> list[int]:
    if args.outer_folds is None:
        return list(range(total))
    folds = [int(x) for x in args.outer_folds.split(",") if x.strip()]
    bad = [x for x in folds if x < 0 or x >= total]
    if bad:
        raise ValueError(f"Outer folds out of range 0..{total - 1}: {bad[:10]}")
    return folds


def main() -> None:
    args = parse_args()
    verbosity = {
        "warning": optuna.logging.WARNING,
        "info": optuna.logging.INFO,
        "debug": optuna.logging.DEBUG,
    }[args.optuna_verbosity]
    optuna.logging.set_verbosity(verbosity)

    tasks = build_tasks(args.screen)
    if args.list_tasks:
        print(f"screen={args.screen}")
        print(f"tasks={len(tasks)}")
        print(f"first={asdict(tasks[0])}")
        print(f"last={asdict(tasks[-1])}")
        return

    id_set = set(selected_task_ids(args, len(tasks)))
    tasks_to_run = [task for task in tasks if task.task_id in id_set]
    print(f"[INFO] screen={args.screen}, running {len(tasks_to_run)} tasks")

    for task in tasks_to_run:
        out_path = task_output_path(args.results_dir, task, args.inner_folds)
        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] task {task.task_id}: {out_path}")
            continue
        print(
            f"[RUN] task {task.task_id}/{len(tasks)} "
            f"{task.descriptor} {task.model}"
        )
        start = time.time()
        try:
            payload = run_task(task, args)
            payload["runtime_sec"] = round(time.time() - start, 3)
            payload["status"] = "ok"
        except Exception as exc:
            payload = {
                "task": asdict(task),
                "status": "failed",
                "error": repr(exc),
                "runtime_sec": round(time.time() - start, 3),
            }
            write_json_atomic(out_path, payload)
            print(f"[FAIL] task {task.task_id}: {exc!r}")
            raise
        write_json_atomic(out_path, payload)
        print(f"[DONE] task {task.task_id}: {out_path}")


if __name__ == "__main__":
    main()
