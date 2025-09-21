import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor as SK_RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor as SK_ExtraTreesRegressor
from sklearn.ensemble import ExtraTreesClassifier as SK_ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier as SK_RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor as SK_GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier as SK_GradientBoostingClassifier
from sklearn.ensemble import AdaBoostRegressor as SK_AdaBoostRegressor
from sklearn.ensemble import AdaBoostClassifier as SK_AdaBoostClassifier
from sklearn.ensemble import BaggingRegressor as SK_BaggingRegressor
from sklearn.ensemble import BaggingClassifier as SK_BaggingClassifier
from sklearn.ensemble import VotingRegressor as SK_VotingRegressor
from sklearn.tree import DecisionTreeRegressor as SK_DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier as SK_DecisionTreeClassifier
from xgboost import XGBRegressor,XGBClassifier
from lightgbm import LGBMRegressor as RAW_LGBMRegressor
from lightgbm import LGBMClassifier as RAW_LGBMClassifier
from sklearn.neural_network import MLPRegressor as SK_MLPRegressor
from sklearn.neural_network import MLPClassifier as SK_MLPClassifier

class RandomForestRegressor(SK_RandomForestRegressor):
    def __init__(self, n_estimators=100, criterion='squared_error', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=1.0, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=-1, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None):
        
        super().__init__(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start, ccp_alpha=ccp_alpha, max_samples=max_samples)

    def save_model(self, filename):
        joblib.dump(self, filename)

    @classmethod
    def load_model(cls, filename):
        return joblib.load(filename)

class ExtraTreesRegressor(SK_ExtraTreesRegressor):
    def __init__(self,n_estimators=100, criterion='squared_error', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=1.0, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=False, oob_score=False, n_jobs=-1, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None):
        
        super().__init__(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start, ccp_alpha=ccp_alpha, max_samples=max_samples)
    
    def save_model(self, filename):
        joblib.dump(self, filename)

    @classmethod
    def load_model(cls, filename):
        return joblib.load(filename)

class MLPRegressor(SK_MLPRegressor):
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, 
    tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, 
    validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000):
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,activation=activation,solver=solver,
            alpha=alpha,batch_size=batch_size,learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,power_t=power_t,
            max_iter=max_iter,shuffle=shuffle,random_state=random_state,
            tol=tol,verbose=verbose,warm_start=warm_start,momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,early_stopping=early_stopping,
            validation_fraction=validation_fraction,beta_1=beta_1,beta_2=beta_2,
            epsilon=epsilon,n_iter_no_change=n_iter_no_change,max_fun=max_fun)

    def save_model(self, filename):
        joblib.dump(self, filename)

    @classmethod
    def load_model(cls, filename):
        return joblib.load(filename)
    
class AdaBoostRegressor(SK_AdaBoostRegressor):
    def __init__(self,estimator=None, n_estimators=50, learning_rate=1.0, loss='linear', random_state=None):
        super().__init__(
            estimator=estimator, n_estimators=n_estimators, learning_rate=learning_rate,
            loss=loss, random_state=random_state)
    def save_model(self, filename):
        joblib.dump(self, filename)

    @classmethod
    def load_model(cls, filename):
        return joblib.load(filename)
    
class BaggingRegressor(SK_BaggingRegressor):
    def __init__(self, estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=-1, random_state=None, verbose=0):
        super().__init__(
            estimator=estimator, n_estimators=n_estimators, max_samples=max_samples,
            max_features=max_features, bootstrap=bootstrap, bootstrap_features=bootstrap_features,
            oob_score=oob_score, warm_start=warm_start, n_jobs=n_jobs, random_state=random_state,
            verbose=verbose)
    def save_model(self, filename):
        joblib.dump(self, filename)

    @classmethod
    def load_model(cls, filename):
        return joblib.load(filename)
    
class VotingRegressor(SK_VotingRegressor):
    def __init__(self, estimators, weights=None, n_jobs=-1, verbose=False):
        super().__init__(estimators=estimators, weights=weights, n_jobs=n_jobs, verbose=verbose)
    def save_model(self, filename):
        joblib.dump(self, filename)

    @classmethod
    def load_model(cls, filename):
        return joblib.load(filename)
    
class DecisionTreeRegressor(SK_DecisionTreeRegressor):
    def __init__(self, criterion='squared_error', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, ccp_alpha=0.0):
        super().__init__(criterion=criterion, splitter=splitter, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, random_state=random_state, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, ccp_alpha=ccp_alpha)
    def save_model(self, filename):
        joblib.dump(self, filename)

    @classmethod
    def load_model(cls, filename):
        return joblib.load(filename)

class DecisionTreeClassifier(SK_DecisionTreeClassifier):
    def __init__(self, *, criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, class_weight=None, ccp_alpha=0.0):
        super().__init__(criterion=criterion, splitter=splitter, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, random_state=random_state, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, class_weight=class_weight, ccp_alpha=ccp_alpha)
    def save_model(self, filename):
        joblib.dump(self, filename)

    @classmethod
    def load_model(cls, filename):
        return joblib.load(filename)

class GradientBoostingRegressor(SK_GradientBoostingRegressor):
    def __init__(self,loss='squared_error', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0):
        super().__init__(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, criterion=criterion, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_depth=max_depth, min_impurity_decrease=min_impurity_decrease, init=init, random_state=random_state, max_features=max_features, alpha=alpha, verbose=verbose, max_leaf_nodes=max_leaf_nodes, warm_start=warm_start, validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change, tol=tol, ccp_alpha=ccp_alpha)
    
    def save_model(self, filename):
        joblib.dump(self, filename)
        
    @classmethod
    def load_model(cls, filename):
        return joblib.load(filename)
    
class RandomForestClassifier(SK_RandomForestClassifier):
    def __init__(self, n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=-1, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None):
        super().__init__(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start, class_weight=class_weight)
    
    def save_model(self, filename):
        joblib.dump(self, filename)
        
    @classmethod
    def load_model(cls, filename):
        return joblib.load(filename)
    
class ExtraTreesClassifier(SK_ExtraTreesClassifier):
    def __init__(self,n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=False, oob_score=False, n_jobs=-1, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None):
        super().__init__(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start, class_weight=class_weight, ccp_alpha=ccp_alpha, max_samples=max_samples)
    
    def save_model(self, filename):
        joblib.dump(self, filename)

    @classmethod
    def load_model(cls, filename):
        return joblib.load(filename)

class MLPClassifier(SK_MLPClassifier):
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', *, solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000):
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,activation=activation,solver=solver,
            alpha=alpha,batch_size=batch_size,learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,power_t=power_t,
            max_iter=max_iter,shuffle=shuffle,random_state=random_state,
            tol=tol,verbose=verbose,warm_start=warm_start,momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,early_stopping=early_stopping,
            validation_fraction=validation_fraction,beta_1=beta_1,beta_2=beta_2,
            epsilon=epsilon,n_iter_no_change=n_iter_no_change,max_fun=max_fun)

    def save_model(self, filename):
        joblib.dump(self, filename)

    @classmethod
    def load_model(cls, filename):
        return joblib.load(filename)

class AdaBoostClassifier(SK_AdaBoostClassifier):
    def __init__(self,estimator=None, *, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None):
        super().__init__(
            estimator=estimator, n_estimators=n_estimators, learning_rate=learning_rate,
            algorithm=algorithm, random_state=random_state)
    def save_model(self, filename):
        joblib.dump(self, filename)

    @classmethod
    def load_model(cls, filename):
        return joblib.load(filename)

class BaggingClassifier(SK_BaggingClassifier):
    def __init__(self, estimator=None, n_estimators=10, *, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=None, random_state=None, verbose=0):
        super().__init__(
            estimator=estimator, n_estimators=n_estimators, max_samples=max_samples,
            max_features=max_features, bootstrap=bootstrap, bootstrap_features=bootstrap_features,
            oob_score=oob_score, warm_start=warm_start, n_jobs=n_jobs, random_state=random_state,
            verbose=verbose)
    def save_model(self, filename):
        joblib.dump(self, filename)

    @classmethod
    def load_model(cls, filename):
        return joblib.load(filename)

class GradientBoostingClassifier(SK_GradientBoostingClassifier):
    def __init__(self, loss='log_loss', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0):
        super().__init__(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, criterion=criterion, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_depth=max_depth, min_impurity_decrease=min_impurity_decrease, init=init, random_state=random_state, max_features=max_features, verbose=verbose, max_leaf_nodes=max_leaf_nodes, warm_start=warm_start, validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change, tol=tol, ccp_alpha=ccp_alpha)

    def save_model(self, filename):
        joblib.dump(self, filename)

    @classmethod
    def load_model(cls, filename):
        return joblib.load(filename)
    
class LGBMRegressor(RAW_LGBMRegressor):
    def __init__(self, boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100, subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20, subsample=1.0, subsample_freq=0, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=None, importance_type='split'):
        
        super().__init__(boosting_type=boosting_type,num_leaves=num_leaves,max_depth=max_depth,learning_rate=learning_rate,n_estimators=n_estimators,
                         subsample_for_bin=subsample_for_bin,objective=objective,class_weight=class_weight,min_split_gain=min_split_gain,
                         min_child_weight=min_child_weight,min_child_samples=min_child_samples,subsample=subsample,subsample_freq=subsample_freq,
                         colsample_bytree=colsample_bytree,reg_alpha=reg_alpha,reg_lambda=reg_lambda,random_state=random_state,n_jobs=n_jobs,
                         importance_type=importance_type,verbosity=-1)
    
    def save_model(self, filename):
        joblib.dump(self, filename)
    
    @classmethod
    def load_model(cls, filename):
        return joblib.load(filename)
    
class LGBMClassifier(RAW_LGBMClassifier):
    def __init__(self,boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100, subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20, subsample=1.0, subsample_freq=0, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=None, importance_type='split'):
        super().__init__(boosting_type=boosting_type,num_leaves=num_leaves,max_depth=max_depth,learning_rate=learning_rate,n_estimators=n_estimators,
                         subsample_for_bin=subsample_for_bin,objective=objective,class_weight=class_weight,min_split_gain=min_split_gain,
                         min_child_weight=min_child_weight,min_child_samples=min_child_samples,subsample=subsample,subsample_freq=subsample_freq,
                          colsample_bytree=colsample_bytree,reg_alpha=reg_alpha,reg_lambda=reg_lambda,random_state=random_state,n_jobs=n_jobs,
                          importance_type=importance_type,verbosity=-1)
    def save_model(self, filename):
        joblib.dump(self, filename)
    
    @classmethod
    def load_model(cls, filename):
        return joblib.load(filename)