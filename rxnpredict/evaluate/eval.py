import numpy as np
from copy import deepcopy
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold,LeaveOneOut,train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,precision_score,recall_score,f1_score,accuracy_score,confusion_matrix,pairwise_distances
def simi_data_select(base_X,dest_X,topk=10,dist_type='euclidean',return_idx=True):
        dist_pair = pairwise_distances(dest_X,base_X,metric=dist_type).sum(axis=0)
        topk = min(topk,base_X.shape[0])
        if isinstance(topk,int):
            simi_data_idx = np.argsort(dist_pair)[:topk]  
        else:
            std_dist_pair = (dist_pair-dist_pair.min())/(dist_pair.max()-dist_pair.min())
            simi_data_idx = np.where(std_dist_pair<topk)[0]  
        if return_idx:
            return simi_data_idx
        else:
            return base_X[simi_data_idx]

def get_predict(model,base_X,base_Y,dest_X,dest_Y,vir_X,merge_method='mix',simi_eval=True,
                dist_type='euclidean',topk=20,base_simi_X=None,dest_simi_X=None,verbose=True,
                return_model=False):
    merge_method = merge_method.lower()
    assert merge_method in ["mix","delta","dest"], "merge_method must be mix, delta or dest"
    if simi_eval:
        if base_simi_X is None:
            base_simi_X = base_X
        if dest_simi_X is None:
            dest_simi_X = dest_X
    if verbose:
        print(f"Evalutaion process: merge method {merge_method}, similarity evaluation {simi_eval}, distance type {dist_type}, topk {topk}")

    
    simi_index_idx_map = {}
    for idx in range(len(vir_X)):
        tmp_vir_X = vir_X[[idx]]
        tmp_simi_vir_X = vir_X[[idx]]
        if simi_eval:
            simi_index = simi_data_select(base_simi_X,tmp_simi_vir_X,topk=topk,dist_type=dist_type,return_idx=True)
            simi_index_tup = tuple(sorted(simi_index))
            if simi_index_tup in simi_index_idx_map:
                simi_index_idx_map[simi_index_tup].append(idx)
            else:
                simi_index_idx_map[simi_index_tup] = [idx]
        else:
            sel_base_X = base_X
            sel_base_Y = base_Y
    vir_P = np.zeros(len(vir_X)) - float("-inf")
    print(f"{len(simi_index_idx_map)} times model fitting")
    for simi_index_tup,idx_lst in simi_index_idx_map.items():
        sel_base_X = base_X[list(simi_index_tup)]
        sel_base_Y = base_Y[list(simi_index_tup)]
        if merge_method == "mix":
            merge_x = np.concatenate([sel_base_X,dest_X])
            merge_y = np.concatenate([sel_base_Y,dest_Y])
            model.fit(merge_x,merge_y)
        elif merge_method == "dest":
            model.fit(dest_X,dest_Y)
        elif merge_method == "delta":
            base_model = deepcopy(model)
            delta_model = deepcopy(model)
            base_model.fit(sel_base_X,sel_base_Y)
            dest_P = base_model.predict(dest_X)
            dest_D = dest_Y - dest_P
            delta_model.fit(dest_X,dest_D)
        if merge_method in ["mix","dest"]:
            vir_P[idx_lst] = model.predict(vir_X[idx_lst])
        elif merge_method in ["delta"]:
            vir_P[idx_lst] = delta_model.predict(vir_X[idx_lst]) + base_model.predict(vir_X[idx_lst])
    if not return_model:
        return vir_P
    else:
        return vir_P,base_model,delta_model

def get_val_score_add_data(model,base_X,base_Y,dest_X,dest_Y,selection_inf,ret_idx=False,merge_method="mix",simi_eval=True,
                           simi_va_base="valid",dist_type='euclidean',topk=20,base_simi_X=None,dest_simi_X=None,verbose=True,
                           return_model=False,ret_feat_importance=False):
    '''
    perform validation and get validation score
    Parameters
    ----------
    model : model object
        ML model
    base_X : ndarray
        numpy array of feature matrix for base data
    base_Y : ndarray
        numpy array of target vector for base data
    dest_X : ndarray
        numpy array of feature matrix for target data
    dest_Y : ndarray
        numpy array of target vector for target data
    selection_inf : dict
        validation information
    ret_idx : bool
        whether return validation index list
    merge_method : str
        merge method for base and target data
    '''
    merge_method = merge_method.lower()
    assert merge_method in ["mix","delta","dest"], "merge_method must be mix, delta or dest"
    assert simi_va_base in ["valid","train"], "simi_va_base must be valid or train"
    if simi_eval:
        if base_simi_X is None:
            base_simi_X = base_X
        if dest_simi_X is None:
            dest_simi_X = dest_X
    if verbose:
        print(f"Evalutaion process: merge method {merge_method}, similarity evaluation {simi_eval}, similarity evaluation based on {simi_va_base}, distance type {dist_type}, topk {topk}")
    val_idx_lst = []
    if selection_inf['type'] == 'cv':
        
        kfold = KFold(n_splits=selection_inf['fold'],shuffle=True,random_state=selection_inf['random_state'],)
        va_Y = []
        va_P = []
        base_feat_importance = []
        delta_feat_importance = []
        for tr_idx,va_idx in kfold.split(dest_X):
            tr_x,tr_y = dest_X[tr_idx],dest_Y[tr_idx]
            va_simi_x = dest_simi_X[va_idx]
            tr_simi_x = dest_simi_X[tr_idx]

            # similarity selection
            if simi_eval:
                if simi_va_base == "valid":
                    simi_index = simi_data_select(base_simi_X,va_simi_x,topk=topk,dist_type=dist_type,return_idx=True)
                elif simi_va_base == "train":
                    simi_index = simi_data_select(base_simi_X,tr_simi_x,topk=topk,dist_type=dist_type,return_idx=True)
                sel_base_X = base_X[simi_index]
                sel_base_Y = base_Y[simi_index]
            else:
                sel_base_X = base_X
                sel_base_Y = base_Y

            if merge_method == "mix":
                merge_x = np.concatenate([sel_base_X,tr_x])
                merge_y = np.concatenate([sel_base_Y,tr_y])
                model.fit(merge_x,merge_y)
            elif merge_method == "dest":
                model.fit(tr_x,tr_y)
            elif merge_method == "delta":
                base_model = deepcopy(model)
                delta_model = deepcopy(model)
                base_model.fit(sel_base_X,sel_base_Y)
                tr_p = base_model.predict(tr_x)
                tr_d = tr_y - tr_p
                delta_model.fit(tr_x,tr_d)
            va_x,va_y = dest_X[va_idx],dest_Y[va_idx]
            if merge_method in ["mix","dest"]:
                va_p = model.predict(va_x)
            elif merge_method in ["delta"]:
                va_p = delta_model.predict(va_x) + base_model.predict(va_x)
                try:
                    base_feat_importance.append(base_model.feature_importances_)
                    delta_feat_importance.append(delta_model.feature_importances_)
                except:
                    base_feat_importance.append(None)
                    delta_feat_importance.append(None)
            va_Y.append(va_y)
            va_P.append(va_p)
            val_idx_lst.append(va_idx)
        va_Y = np.concatenate(va_Y)
        va_P = np.concatenate(va_P)
        val_idx_lst = np.concatenate(val_idx_lst)
        score_map = {}
        if 'r2' in selection_inf['metric']:
            score_map['r2'] = r2_score(va_Y,va_P)
        if 'mae' in selection_inf['metric']:
            score_map['mae'] = mean_absolute_error(va_Y,va_P)
        if 'mse' in selection_inf['metric'] :
            score_map['mse'] = mean_squared_error(va_Y,va_P)
        if 'precision' in selection_inf['metric']:
            score_map['precision'] = precision_score(va_Y,va_P)
        if 'recall' in selection_inf['metric']:
            score_map['recall'] = recall_score(va_Y,va_P)
        if 'f1' in selection_inf['metric']:
            score_map['f1'] = f1_score(va_Y,va_P)
        if 'accuracy' in selection_inf['metric']:
            score_map['accuracy'] = accuracy_score(va_Y,va_P)
        if 'confusion_matrix' in selection_inf['metric']:
            score_map['confusion_matrix'] = confusion_matrix(va_Y,va_P)
        if len(score_map) == 0:
            raise Exception('metric error')
       
    elif selection_inf['type'] == 'loo':
        loo = LeaveOneOut()
        va_Y = []
        va_P = []
        base_feat_importance = []
        delta_feat_importance = []
        for tr_idx,va_idx in loo.split(dest_X):
            tr_x,tr_y = dest_X[tr_idx],dest_Y[tr_idx]
            va_simi_x = dest_simi_X[va_idx]
            tr_simi_x = dest_simi_X[tr_idx]
            # similarity selection
            if simi_eval:
                if simi_va_base == "valid":
                    simi_index = simi_data_select(base_simi_X,va_simi_x,topk=topk,dist_type=dist_type,return_idx=True)
                elif simi_va_base == "train":
                    simi_index = simi_data_select(base_simi_X,tr_simi_x,topk=topk,dist_type=dist_type,return_idx=True)
                sel_base_X = base_X[simi_index]
                sel_base_Y = base_Y[simi_index]
            else:
                sel_base_X = base_X
                sel_base_Y = base_Y

            if merge_method == "mix":
                merge_x = np.concatenate([sel_base_X,tr_x])
                merge_y = np.concatenate([sel_base_Y,tr_y])
                model.fit(merge_x,merge_y)
            elif merge_method == "dest":
                model.fit(tr_x,tr_y)
            elif merge_method == "delta":
                base_model = deepcopy(model)
                delta_model = deepcopy(model)
                base_model.fit(sel_base_X,sel_base_Y)
                tr_p = base_model.predict(tr_x)
                tr_d = tr_y - tr_p
                delta_model.fit(tr_x,tr_d)
                try:
                    base_feat_importance.append(base_model.feature_importances_)
                    delta_feat_importance.append(delta_model.feature_importances_)
                except:
                    base_feat_importance.append(None)
                    delta_feat_importance.append(None)
            va_x,va_y = dest_X[va_idx],dest_Y[va_idx]
            if merge_method in ["mix","dest"]:
                va_p = model.predict(va_x)
            elif merge_method in ["delta"]:
                va_p = delta_model.predict(va_x) + base_model.predict(va_x)
            va_Y.append(va_y)
            va_P.append(va_p)
            val_idx_lst.append(va_idx)
        va_Y = np.concatenate(va_Y)
        va_P = np.concatenate(va_P)
        val_idx_lst = np.concatenate(val_idx_lst)
        score_map = {}
        if 'r2' in selection_inf['metric']:
            score_map['r2'] = r2_score(va_Y,va_P)
        if 'mae' in selection_inf['metric']:
            score_map['mae'] = mean_absolute_error(va_Y,va_P)
        if 'mse' in selection_inf['metric'] :
            score_map['mse'] = mean_squared_error(va_Y,va_P)
        if 'precision' in selection_inf['metric']:
            score_map['precision'] = precision_score(va_Y,va_P)
        if 'recall' in selection_inf['metric']:
            score_map['recall'] = recall_score(va_Y,va_P)
        if 'f1' in selection_inf['metric']:
            score_map['f1'] = f1_score(va_Y,va_P)
        if 'accuracy' in selection_inf['metric']:
            score_map['accuracy'] = accuracy_score(va_Y,va_P)
        if 'confusion_matrix' in selection_inf['metric']:
            score_map['confusion_matrix'] = confusion_matrix(va_Y,va_P)
        if len(score_map) == 0:
            raise Exception('metric error')
    
    else:
        tr_x,tr_y,va_x,va_y = train_test_split(dest_X,dest_Y,test_size=selection_inf['test_size'],random_state=selection_inf['random_state'])
        tr_simi_x,va_simi_x = train_test_split(dest_simi_X,test_size=selection_inf['test_size'],random_state=selection_inf['random_state'])

        # similarity selection
        if simi_eval:
            if simi_va_base == "valid":
                simi_index = simi_data_select(base_simi_X,va_simi_x,topk=topk,dist_type=dist_type,return_idx=True)
            elif simi_va_base == "train":
                simi_index = simi_data_select(base_simi_X,tr_simi_x,topk=topk,dist_type=dist_type,return_idx=True)
            sel_base_X = base_X[simi_index]
            sel_base_Y = base_Y[simi_index]
        else:
            sel_base_X = base_X
            sel_base_Y = base_Y

        if merge_method == "mix":
            merge_x = np.concatenate([sel_base_X,tr_x])
            merge_y = np.concatenate([sel_base_Y,tr_y])
            model.fit(merge_x,merge_y)
        elif merge_method == "dest":
            model.fit(tr_x,tr_y)
        elif merge_method == "delta":
            base_model = deepcopy(model)
            delta_model = deepcopy(model)
            base_model.fit(sel_base_X,sel_base_Y)
            tr_p = base_model.predict(tr_x)
            tr_d = tr_y - tr_p
            delta_model.fit(tr_x,tr_d)
        if merge_method in ["mix","dest"]:
            va_p = model.predict(va_x)
        elif merge_method in ["delta"]:
            va_p = delta_model.predict(va_x) + base_model.predict(va_x)
            
        va_Y = deepcopy(va_y)
        va_P = deepcopy(va_p)
        
        score_map = {}
        if 'r2' in selection_inf['metric']:
            score_map['r2'] = r2_score(va_Y,va_P)
        if 'mae' in selection_inf['metric']:
            score_map['mae'] = mean_absolute_error(va_Y,va_P)
        if 'mse' in selection_inf['metric'] :
            score_map['mse'] = mean_squared_error(va_Y,va_P)
        if 'precision' in selection_inf['metric']:
            score_map['precision'] = precision_score(va_Y,va_P)
        if 'recall' in selection_inf['metric']:
            score_map['recall'] = recall_score(va_Y,va_P)
        if 'f1' in selection_inf['metric']:
            score_map['f1'] = f1_score(va_Y,va_P)
        if 'accuracy' in selection_inf['metric']:
            score_map['accuracy'] = accuracy_score(va_Y,va_P)
        if 'confusion_matrix' in selection_inf['metric']:
            score_map['confusion_matrix'] = confusion_matrix(va_Y,va_P)
        if len(score_map) == 0:
            raise Exception('metric error')
    if ret_idx:
        if ret_feat_importance and merge_method == "delta":
            return va_Y,va_P,score_map,val_idx_lst,base_feat_importance,delta_feat_importance
        else:
            return va_Y,va_P,score_map,val_idx_lst
    else:
        if ret_feat_importance and merge_method == "delta":
            return va_Y,va_P,score_map,base_feat_importance,delta_feat_importance
        else:
            return va_Y,va_P,score_map