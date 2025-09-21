from .ml import *
from copy import deepcopy
import numpy as np

REGRESSOR_LIST = [
    'RandomForestRegressor',
    'ExtraTreesRegressor',
    'MLPRegressor',
    'AdaBoostRegressor',
    'BaggingRegressor',
    'DecisionTreeRegressor',
    'GradientBoostingRegressor',
    'XGBRegressor',
    'LGBMRegressor',
]
CLASSIFIER_LIST = [
    'RandomForestClassifier',
    'ExtraTreesClassifier',
    'GradientBoostingClassifier',
    'MLPClassifier',
    'AdaBoostClassifier',
    'BaggingClassifier',
    'DecisionTreeClassifier',
    'XGBClassifier',
    'LGBMClassifier',
]
def get_model(model_name="RandomForestRegressor",model_params={}):
    if model_name == "RandomForestRegressor":
        return RandomForestRegressor(**model_params)
    elif model_name == "ExtraTreesRegressor":
        return ExtraTreesRegressor(**model_params)
    elif model_name == "MLPRegressor":
        return MLPRegressor(**model_params)
    elif model_name == "AdaBoostRegressor":
        return AdaBoostRegressor(**model_params)
    elif model_name == 'BaggingRegressor':
        return BaggingRegressor(**model_params)
    elif model_name == 'XGBRegressor':
        return XGBRegressor(**model_params)
    elif model_name == 'LGBMRegressor':
        return LGBMRegressor(**model_params)
    elif model_name == 'DecisionTreeRegressor':
        return DecisionTreeRegressor(**model_params)
    elif model_name == 'GradientBoostingRegressor':
        return GradientBoostingRegressor(**model_params)
    elif model_name == 'RandomForestClassifier':
        return RandomForestClassifier(**model_params)
    elif model_name == 'ExtraTreesClassifier':
        return ExtraTreesClassifier(**model_params)
    elif model_name == 'GradientBoostingClassifier':
        return GradientBoostingClassifier(**model_params)
    elif model_name == "MLPClassifier":
        return MLPClassifier(**model_params)
    elif model_name == "AdaBoostClassifier":
        return AdaBoostClassifier(**model_params)
    elif model_name == 'BaggingClassifier':
        return BaggingClassifier(**model_params)
    elif model_name == 'DecisionTreeClassifier':
        return DecisionTreeClassifier(**model_params)
    elif model_name == 'XGBClassifier':
        return XGBClassifier(**model_params)
    elif model_name == 'LGBMClassifier':
        return LGBMClassifier(**model_params)
    else:
        raise ValueError("Model name not recognized")
