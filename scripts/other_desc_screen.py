
from sklearn.ensemble import ExtraTreesRegressor,AdaBoostRegressor,BaggingRegressor,GradientBoostingRegressor
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
from rxnpredict.evaluate.eval import get_val_score_add_data
import optuna,argparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from copy import deepcopy
from sklearn.metrics import r2_score
warnings.filterwarnings("ignore")


def rf_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 6),
        'max_depth': trial.suggest_int('max_depth', 3, 60),
    }
    model = RandomForestRegressor(**params,random_state=random_state,n_jobs=-1)
    va_Y,va_P,score_map = get_val_score_add_data(model,base_X,base_y,X,y,
                                                 selection_inf=selection_inf,merge_method=merge_method,topk=300,dist_type=dist_type)
    
    return score_map['r2']

def et_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 6),
        'max_depth': trial.suggest_int('max_depth', 3, 60),
    }
    model = ExtraTreesRegressor(**params,random_state=random_state,n_jobs=-1)

    va_Y,va_P,score_map = get_val_score_add_data(model,base_X,base_y,X,y,
                                                 selection_inf=selection_inf,merge_method=merge_method,topk=300,dist_type=dist_type)
    return score_map['r2']

def mlp_objective(trial):
    params = {
        'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(50,),
                                                                               (100,),
                                                                               (50, 100, 50), 
                                                                               (100, 50, 100)]),
        'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
        'learning_rate_init': trial.suggest_categorical('learning_rate_init', [0.0001, 0.001, 0.01]),
    }
    model = MLPRegressor(**params,random_state=random_state)

    va_Y,va_P,score_map = get_val_score_add_data(model,base_X,base_y,X,y,
                                                 selection_inf=selection_inf,merge_method=merge_method,topk=300,dist_type=dist_type)
    return score_map['r2']

def ada_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.01, 0.1, 1.0]),
    }
    model = AdaBoostRegressor(**params,random_state=random_state)

    va_Y,va_P,score_map = get_val_score_add_data(model,base_X,base_y,X,y,
                                                 selection_inf=selection_inf,merge_method=merge_method,topk=300,dist_type=dist_type)
    return score_map['r2']

def bg_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 100),
        'max_samples': trial.suggest_categorical('max_samples', [0.1, 0.5, 1.0]),
        'max_features': trial.suggest_categorical('max_features', [0.1, 0.5, 1.0]),
    }
    model = BaggingRegressor(**params,random_state=random_state,n_jobs=-1)

    va_Y,va_P,score_map = get_val_score_add_data(model,base_X,base_y,X,y,
                                                 selection_inf=selection_inf,merge_method=merge_method,topk=300,dist_type=dist_type)
    return score_map['r2']

def dt_objective(trial):
    params = {
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_depth': trial.suggest_int('max_depth', 3, 60),
    }
    model = DecisionTreeRegressor(**params,random_state=random_state)

    va_Y,va_P,score_map = get_val_score_add_data(model,base_X,base_y,X,y,
                                                 selection_inf=selection_inf,merge_method=merge_method,topk=300,dist_type=dist_type)
    return score_map['r2']

def gb_objective(trial):
    params = {
         'n_estimators': trial.suggest_int('n_estimators', 50, 300),
         'min_samples_split': trial.suggest_int('min_samples_split', 2, 6),
        'max_depth': trial.suggest_int('max_depth', 3, 60),
    }
    model = GradientBoostingRegressor(**params,random_state=random_state)
    va_Y,va_P,score_map = get_val_score_add_data(model,base_X,base_y,X,y,
                                                 selection_inf=selection_inf,merge_method=merge_method,topk=300,dist_type=dist_type)
    return score_map['r2']

def xgb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 6),
        'max_depth': trial.suggest_int('max_depth', 3, 60),
    }
    model = XGBRegressor(**params,random_state=random_state,n_jobs=-1)
    va_Y,va_P,score_map = get_val_score_add_data(model,base_X,base_y,X,y,
                                                 selection_inf=selection_inf,merge_method=merge_method,topk=300,dist_type=dist_type)
    return score_map['r2']

def lgbm_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'num_leaves': trial.suggest_int('num_leaves', 31, 60),
    }
    model = LGBMRegressor(**params,random_state=random_state,n_jobs=-1)
    va_Y,va_P,score_map = get_val_score_add_data(model,base_X,base_y,X,y,
                                                 selection_inf=selection_inf,merge_method=merge_method,topk=300,dist_type=dist_type)
    return score_map['r2']




merge_method = "delta"
dist_type = "euclidean"
random_state = 1024
n_estimators = 50
descriptor_target_pearsonr_threshold = 0.2
descriptor_pearsonr_threshold = 0.95
base_rct_keys = ["Reactant1","Reactant2"]
target_rct_keys = ["Reactant1","Reactant2"]
base_sol_keys = ["Solvents"]
target_sol_keys = ["Solvents"]
base_rgt_keys = ["Reagents"]
target_rgt_keys = ["Reagents"]
base_ts_key = ["Reactant2","Reagent5"]
target_ts_key = ["Reactant2","Reagent2"]
selection_inf = {
            "type": "loo",
            "fold": 10,
            "metric": ["r2","mae"]}


final_descriptor_map = np.load("../desc/final_descriptor_map.npy", allow_pickle=True).item()

parser = argparse.ArgumentParser()
parser.add_argument('--n_trials', type=int, default=15)
parser.add_argument('--desc_name', type=str, default="SPOC")
args = parser.parse_args()
print(f"Descriptor Name: {args.desc_name}")
n_trials = args.n_trials
desc_name = args.desc_name


desc_map = final_descriptor_map[desc_name]
X = desc_map['target_X']
y = desc_map['target_y']
base_X = desc_map['base_X']
base_y = desc_map['base_y']

for model_name in ["gb","rf","et","mlp","ada","bg","dt","xgb","lgbm"]:
    sampler = TPESampler(n_startup_trials=5, seed=random_state)
    study = optuna.create_study(direction='maximize',sampler=sampler)
    if model_name == 'rf':
        study.optimize(rf_objective, n_trials=n_trials)
        best_model = RandomForestRegressor(**study.best_params, random_state=random_state)
    elif model_name == 'et':
        study.optimize(et_objective, n_trials=n_trials)
        best_model = ExtraTreesRegressor(**study.best_params, random_state=random_state)
    elif model_name == 'mlp':
        study.optimize(mlp_objective, n_trials=n_trials)
        best_model = MLPRegressor(**study.best_params, random_state=random_state)
    elif model_name == 'ada':
        study.optimize(ada_objective, n_trials=n_trials)
        best_model = AdaBoostRegressor(**study.best_params, random_state=random_state)
    elif model_name == 'bg':
        study.optimize(bg_objective, n_trials=n_trials)
        best_model = BaggingRegressor(**study.best_params, random_state=random_state)
    elif model_name == 'dt':
        study.optimize(dt_objective, n_trials=n_trials)
        best_model = DecisionTreeRegressor(**study.best_params)
    elif model_name == 'gb':
        study.optimize(gb_objective, n_trials=n_trials)
        best_model = GradientBoostingRegressor(**study.best_params, random_state=random_state)
    elif model_name == 'xgb':
        study.optimize(xgb_objective, n_trials=n_trials)
        best_model = XGBRegressor(**study.best_params, random_state=random_state)
    elif model_name == 'lgbm':
        study.optimize(lgbm_objective, n_trials=n_trials)
        best_model = LGBMRegressor(**study.best_params, random_state=random_state)

    va_Y,va_P,score_map = get_val_score_add_data(best_model,base_X,base_y,X,y,
                                selection_inf=selection_inf,merge_method=merge_method,topk=300,dist_type=dist_type)
    print(f"[INFO] !!!!!!!!!!!!! {model_name.upper()} {desc_name} R2: {score_map['r2']:.4f} !!!!!!!!!!!!!")
    
    np.save(f"./results/{desc_name}_{model_name}_{score_map['r2']}.npy",score_map)



