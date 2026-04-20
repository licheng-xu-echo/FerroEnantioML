from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,AdaBoostRegressor,BaggingRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from optuna.samplers import TPESampler
import pandas as pd
import numpy as np
import warnings
from rxnpredict.ts.desc import sel_satisfy_ts_data,vec_ts_pair_in_dataset,get_diff_ts_desc
from rxnpredict.descriptors.desc import sel_high_corr_desc,reduce_desc_with_corr_matrix
from rxnpredict.descriptors.utils import process_desc,maxminscale
from rxnpredict.models.utils import get_model
from rxnpredict.models.hyperopt import rf_objective,et_objective,mlp_objective,ada_objective,bg_objective,dt_objective,gb_objective,xgb_objective,lgbm_objective
from rxnpredict.evaluate.eval import get_val_score_add_data
import optuna,argparse,glob
from sklearn.model_selection import LeaveOneOut
from copy import deepcopy
from sklearn.metrics import r2_score
from functools import partial

warnings.filterwarnings("ignore")

random_state = 1024
topk = 300
merge_method = "delta"
dist_type = "euclidean"
selection_inf = {
            "type": "loo",
            "fold": 10,
            "metric": ["r2","mae"]}

final_descriptor_ferrocene_ligand = np.load("../desc/final_descriptor_ferrocene_ligand.npy", allow_pickle=True).item()
parser = argparse.ArgumentParser()
parser.add_argument('--n_trials', type=int, default=15)
parser.add_argument('--desc_name', type=str, default="SPOC")
args = parser.parse_args()
print(f"Descriptor Name: {args.desc_name}")
n_trials = args.n_trials
desc_name = args.desc_name


desc_map = final_descriptor_ferrocene_ligand[desc_name]
X = desc_map['target_X']
y = desc_map['target_y']
base_X = desc_map['base_X']
base_y = desc_map['base_y']

for model_name in ["gb","rf","et","mlp","ada","bg","dt","xgb","lgbm"]:
    res = glob.glob(f"./results/ferr_lig_{desc_name}_{model_name}_*.npy")
    if len(res) > 0:
        print(f"{desc_name} Model {model_name} already exists")
        continue
    sampler = TPESampler(n_startup_trials=5, seed=random_state)
    study = optuna.create_study(direction='maximize',sampler=sampler)
    if model_name == 'rf':
        optimize_func = partial(rf_objective, base_X=base_X,base_y=base_y,X=X,y=y,
                                selection_inf=selection_inf,random_state=random_state,
                                merge_method=merge_method,topk=topk,dist_type=dist_type)
        study.optimize(optimize_func, n_trials=n_trials)
        best_model = RandomForestRegressor(**study.best_params, random_state=random_state)
    elif model_name == 'et':
        optimize_func = partial(et_objective, base_X=base_X,base_y=base_y,X=X,y=y,
                                selection_inf=selection_inf,random_state=random_state,
                                merge_method=merge_method,topk=topk,dist_type=dist_type)
        study.optimize(optimize_func, n_trials=n_trials)
        best_model = ExtraTreesRegressor(**study.best_params, random_state=random_state)
    elif model_name == 'mlp':
        optimize_func = partial(mlp_objective, base_X=base_X,base_y=base_y,X=X,y=y,
                                selection_inf=selection_inf,random_state=random_state,
                                merge_method=merge_method,topk=topk,dist_type=dist_type)
        study.optimize(optimize_func, n_trials=n_trials)
        best_model = MLPRegressor(**study.best_params, random_state=random_state)
    elif model_name == 'ada':
        optimize_func = partial(ada_objective, base_X=base_X,base_y=base_y,X=X,y=y,
                                selection_inf=selection_inf,random_state=random_state,
                                merge_method=merge_method,topk=topk,dist_type=dist_type)
        study.optimize(optimize_func, n_trials=n_trials)
        best_model = AdaBoostRegressor(**study.best_params, random_state=random_state)
    elif model_name == 'bg':
        optimize_func = partial(bg_objective, base_X=base_X,base_y=base_y,X=X,y=y,
                                selection_inf=selection_inf,random_state=random_state,
                                merge_method=merge_method,topk=topk,dist_type=dist_type)
        study.optimize(optimize_func, n_trials=n_trials)
        best_model = BaggingRegressor(**study.best_params, random_state=random_state)
    elif model_name == 'dt':
        optimize_func = partial(dt_objective, base_X=base_X,base_y=base_y,X=X,y=y,
                                selection_inf=selection_inf,random_state=random_state,
                                merge_method=merge_method,topk=topk,dist_type=dist_type)
        study.optimize(optimize_func, n_trials=n_trials)
        best_model = DecisionTreeRegressor(**study.best_params)
    elif model_name == 'gb':
        optimize_func = partial(gb_objective, base_X=base_X,base_y=base_y,X=X,y=y,
                                selection_inf=selection_inf,random_state=random_state,
                                merge_method=merge_method,topk=topk,dist_type=dist_type)
        study.optimize(optimize_func, n_trials=n_trials)
        best_model = GradientBoostingRegressor(**study.best_params, random_state=random_state)
    elif model_name == 'xgb':
        optimize_func = partial(xgb_objective, base_X=base_X,base_y=base_y,X=X,y=y,
                                selection_inf=selection_inf,random_state=random_state,
                                merge_method=merge_method,topk=topk,dist_type=dist_type)
        study.optimize(optimize_func, n_trials=n_trials)
        best_model = XGBRegressor(**study.best_params, random_state=random_state)
    elif model_name == 'lgbm':
        optimize_func = partial(lgbm_objective, base_X=base_X,base_y=base_y,X=X,y=y,
                                selection_inf=selection_inf,random_state=random_state,
                                merge_method=merge_method,topk=topk,dist_type=dist_type)
        study.optimize(optimize_func, n_trials=n_trials)
        best_model = LGBMRegressor(**study.best_params, random_state=random_state)

    va_Y,va_P,score_map = get_val_score_add_data(best_model,base_X,base_y,X,y,
                                selection_inf=selection_inf,merge_method=merge_method,topk=topk,dist_type=dist_type)
    print(f"[INFO] !!!!!!!!!!!!! {model_name.upper()} {desc_name} R2: {score_map['r2']:.4f} !!!!!!!!!!!!!")
    
    np.save(f"./results/ferr_lig_{desc_name}_{model_name}_{score_map['r2']}.npy",score_map)
