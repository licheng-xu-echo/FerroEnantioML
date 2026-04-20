from sklearn.ensemble import ExtraTreesRegressor,AdaBoostRegressor,BaggingRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from optuna.samplers import TPESampler
from ..evaluate.eval import get_val_score_add_data

def rf_objective(trial,base_X,base_y,X,y,selection_inf,random_state=1024,merge_method='delta',topk=300,dist_type='euclidean'):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 6),
        'max_depth': trial.suggest_int('max_depth', 3, 60),
    }
    model = RandomForestRegressor(**params,random_state=random_state,n_jobs=-1)
    va_Y,va_P,score_map = get_val_score_add_data(model,base_X,base_y,X,y,
                                                 selection_inf=selection_inf,merge_method=merge_method,topk=topk,dist_type=dist_type)
    
    return score_map['r2']

def et_objective(trial,base_X,base_y,X,y,selection_inf,random_state=1024,merge_method='delta',topk=300,dist_type='euclidean'):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 6),
        'max_depth': trial.suggest_int('max_depth', 3, 60),
    }
    model = ExtraTreesRegressor(**params,random_state=random_state,n_jobs=-1)

    va_Y,va_P,score_map = get_val_score_add_data(model,base_X,base_y,X,y,
                                                 selection_inf=selection_inf,merge_method=merge_method,topk=topk,dist_type=dist_type)
    return score_map['r2']

def mlp_objective(trial,base_X,base_y,X,y,selection_inf,random_state=1024,merge_method='delta',topk=300,dist_type='euclidean'):
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
                                                 selection_inf=selection_inf,merge_method=merge_method,topk=topk,dist_type=dist_type)
    return score_map['r2']

def ada_objective(trial,base_X,base_y,X,y,selection_inf,random_state=1024,merge_method='delta',topk=300,dist_type='euclidean'):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.01, 0.1, 1.0]),
    }
    model = AdaBoostRegressor(**params,random_state=random_state)

    va_Y,va_P,score_map = get_val_score_add_data(model,base_X,base_y,X,y,
                                                 selection_inf=selection_inf,merge_method=merge_method,topk=topk,dist_type=dist_type)
    return score_map['r2']

def bg_objective(trial,base_X,base_y,X,y,selection_inf,random_state=1024,merge_method='delta',topk=300,dist_type='euclidean'):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 100),
        'max_samples': trial.suggest_categorical('max_samples', [0.1, 0.5, 1.0]),
        'max_features': trial.suggest_categorical('max_features', [0.1, 0.5, 1.0]),
    }
    model = BaggingRegressor(**params,random_state=random_state,n_jobs=-1)

    va_Y,va_P,score_map = get_val_score_add_data(model,base_X,base_y,X,y,
                                                 selection_inf=selection_inf,merge_method=merge_method,topk=topk,dist_type=dist_type)
    return score_map['r2']

def dt_objective(trial,base_X,base_y,X,y,selection_inf,random_state=1024,merge_method='delta',topk=300,dist_type='euclidean'):
    params = {
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_depth': trial.suggest_int('max_depth', 3, 60),
    }
    model = DecisionTreeRegressor(**params,random_state=random_state)

    va_Y,va_P,score_map = get_val_score_add_data(model,base_X,base_y,X,y,
                                                 selection_inf=selection_inf,merge_method=merge_method,topk=topk,dist_type=dist_type)
    return score_map['r2']

def gb_objective(trial,base_X,base_y,X,y,selection_inf,random_state=1024,merge_method='delta',topk=300,dist_type='euclidean'):
    params = {
         'n_estimators': trial.suggest_int('n_estimators', 50, 300),
         'min_samples_split': trial.suggest_int('min_samples_split', 2, 6),
        'max_depth': trial.suggest_int('max_depth', 3, 60),
    }
    model = GradientBoostingRegressor(**params,random_state=random_state)
    va_Y,va_P,score_map = get_val_score_add_data(model,base_X,base_y,X,y,
                                                 selection_inf=selection_inf,merge_method=merge_method,topk=topk,dist_type=dist_type)
    return score_map['r2']

def xgb_objective(trial,base_X,base_y,X,y,selection_inf,random_state=1024,merge_method='delta',topk=300,dist_type='euclidean'):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 6),
        'max_depth': trial.suggest_int('max_depth', 3, 60),
    }
    model = XGBRegressor(**params,random_state=random_state,n_jobs=-1)
    va_Y,va_P,score_map = get_val_score_add_data(model,base_X,base_y,X,y,
                                                 selection_inf=selection_inf,merge_method=merge_method,topk=topk,dist_type=dist_type)
    return score_map['r2']

def lgbm_objective(trial,base_X,base_y,X,y,selection_inf,random_state=1024,merge_method='delta',topk=300,dist_type='euclidean'):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'num_leaves': trial.suggest_int('num_leaves', 31, 60),
    }
    model = LGBMRegressor(**params,random_state=random_state,n_jobs=-1)
    va_Y,va_P,score_map = get_val_score_add_data(model,base_X,base_y,X,y,
                                                 selection_inf=selection_inf,merge_method=merge_method,topk=topk,dist_type=dist_type)
    return score_map['r2']


