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
from rxnpredict.models.hyperopt import rf_objective,et_objective,mlp_objective,ada_objective,bg_objective,dt_objective,gb_objective,xgb_objective,lgbm_objective
from rxnpredict.evaluate.eval import get_val_score_add_data
import optuna,argparse,glob
from sklearn.ensemble import RandomForestRegressor
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

final_descriptor_ferrocene_ligand = np.load("../desc/final_descriptor_map.npy", allow_pickle=True).item()


desc_map = final_descriptor_ferrocene_ligand['SPOC']
X = desc_map['target_X']
y = desc_map['target_y']
base_X = desc_map['base_X']
base_y = desc_map['base_y']
model = GradientBoostingRegressor(n_estimators=117, min_samples_split=5, max_depth=3, random_state=1024)
tmp_va_Y,tmp_va_P,best_score_map = get_val_score_add_data(model,base_X, base_y,
                                                     X, y,
                                                     selection_inf=selection_inf,merge_method=merge_method,topk=topk,dist_type=dist_type)
## feature selection
droped_feat_idx_lst = []
droped_idx_score_map = {}
remain_feat_idx = np.arange(base_X.shape[1])
print(f"R2: {best_score_map['r2']:.4f}")
while True:
    cur_droped_feat_num = len(droped_feat_idx_lst)
    for idx in remain_feat_idx:
        tmp_droped_feat_idx_lst = droped_feat_idx_lst + [idx]
        tmp_remain_feat_idx = [idx for idx in remain_feat_idx if idx not in tmp_droped_feat_idx_lst]
        tmp_va_Y,tmp_va_P,tmp_score_map = get_val_score_add_data(model,base_X[:,tmp_remain_feat_idx], base_y,
                                                     X[:,tmp_remain_feat_idx], y,
                                                     selection_inf=selection_inf,merge_method=merge_method,topk=topk,dist_type=dist_type)
        if tmp_score_map['r2'] > best_score_map['r2']:
            best_score_map = tmp_score_map
            droped_feat_idx_lst = tmp_droped_feat_idx_lst
            remain_feat_idx = tmp_remain_feat_idx
            print(f"R2: {best_score_map['r2']:.4f}", droped_feat_idx_lst)
            droped_idx_score_map[tuple(droped_feat_idx_lst)] = best_score_map['r2']
            break
    if cur_droped_feat_num == len(droped_feat_idx_lst):
        break
np.save(f"./results/dropped_feat_idx_{best_score_map['r2']}.npy",droped_feat_idx_lst)
np.save(f"./results/dropped_idx_score_map_{best_score_map['r2']}.npy",droped_idx_score_map)