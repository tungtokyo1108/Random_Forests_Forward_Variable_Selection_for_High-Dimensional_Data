# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 13:37:32 2020

@author: biomet
"""

import numbers
from warnings import catch_warnings, simplefilter, warn
import threading

from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.sparse import issparse
from scipy.sparse import hstack as sparce_hstack
from joblib import Parallel, delayed

from sklearn.base import ClassifierMixin, RegressorMixin, MultiOutputMixin
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._tree import DTYPE, DOUBLE
from sklearn.utils import check_random_state, check_array, compute_sample_weight
from sklearn.exceptions import DataConversionWarning
from sklearn.ensemble._base import BaseEnsemble, _partition_estimators
from sklearn.utils.fixes import _joblib_parallel_args
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, _check_sample_weight
from sklearn.utils.validation import _deprecate_positional_args


MAX_INT = np.iinfo(np.int32).max

def _get_n_samples_bootstrap(n_samples, max_samples):
    """
    Get the number of samples in a bootstraping sample

    Parameters
    ----------
    n_samples : 
        Number of samples in the dataset.
    max_samples : 
        The maximum number of samples to draw from the total available samples.

    Returns
    -------

    """
    
    if max_samples is None:
        max_samples = n_samples
        
    if isinstance(max_samples, numbers.Integral):
        return max_samples
    
    if isinstance(max_samples, numbers.Real):
        return int(round(n_samples * max_samples))

def _generate_sample_indices(random_state, n_samples, n_samples_bootstrap):
    
    """
    Private function used to run _parallel_build_trees
    """
    
    random_instance = check_random_state(random_state)
    sample_indices = random_instance.randint(0, n_samples, n_samples_bootstrap)
    
    return sample_indices

def _generate_unsampled_indices(random_state, n_samples, n_samples_bootstrap):
    
    sample_indices = _generate_sample_indices(random_state, n_samples, n_samples_bootstrap)
    sample_counts = np.bincount(sample_indices, minlength=n_samples)
    unsampled_mask = sample_counts == 0
    indices_range = np.arange(n_samples)
    unsampled_indices = indices_range[unsampled_mask]
    
    return unsampled_indices

def _parallel_build_trees(tree, forest, X, y, sample_weight, tree_idx, n_trees, 
                          verbose = 0, class_weight = None, 
                          n_samples_bootstrap = None):
    
    if forest.bootstrap:
        n_samples = X.shape[0]
        if sample_weight is None:
            curr_sample_weight = np.ones((n_samples,), dtype = np.float64)
        else:
            curr_sample_weight = sample_weight.copy()
            
        indices = _generate_sample_indices(tree.random_state, 
                                           n_samples, n_samples_bootstrap)
        sample_counts = np.bincount(indices, minlength = n_samples)
        curr_sample_weight *= sample_counts
        
        if class_weight == 'subsample':
            with catch_warnings():
                simplefilter('ignore', DeprecationWarning)
                curr_sample_weight *= compute_sample_weight('auto', y, 
                                                            indices = indices)
        elif class_weight == 'balanced_subsample':
            curr_sample_weight *= compute_sample_weight('balanced', y, 
                                                        indices = indices)
            
        tree.fit(X, y, sample_weight = curr_sample_weight, check_input = False)
    else:
        tree.fit(X, y, sample_weight = sample_weight, check_input = False)
    
    return tree

class BaseForest(MultiOutputMixin, BaseEnsemble, metaclass = ABCMeta):
    
    @abstractmethod
    def __init__(self,
                 base_estimator,
                 n_estimators=100, *,
                 estimator_params=tuple(),
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None,
                 max_samples=None):
        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params)

        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.max_samples = max_samples
    
    def fit(self, X, y, sample_weight = None):
        
        X, y = self._validate_data(X, y, multi_output = True, 
                                   accept_sparse = "csc", dtype = DTYPE)
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)
        
        if issparse(X):
            X.sort_indices()
        
        self.n_features_ = X.shape[1]
        
        y = np.atleast_1d(y)
        
        if y.ndim == 1:
            y = np.reshape(y, (-1,1))
            
        self.n_outputs_ = y.shape[1]
        
        y, expanded_class_weight = self._validate_y_class_weight(y)
        
        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contigous:
            y = np.ascontiguousarray(y, dtype = DOUBLE)
            
        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight
        
        n_samples_bootstrap = _get_n_samples_bootstrap(n_samples = X.shape[0],
                                                       max_samples = self.max_samples)
        
        self._validate_estimator()
        
        random_state = check_random_state(self.random_state)
        
        if not self.warm_start or not hasattr(self, "estimators_"):
            self.estimators_ = []
            
        n_more_estimators = self.n_estimators - len(self.estimators_)
        
        if self.warm_start and len(self.estimators_) > 0:
            random_state.randint(MAX_INT, size = len(self.estimators_))
        
        trees = [self._make_estimator(append=False, random_state = random_state)
                 for i in range(n_more_estimators)]
        
        trees = Parallel(n_jobs = -1, verbose = self.verbose, 
                         **_joblib_parallel_args(prefer='threads'))(
                delayed(_parallel_build_trees)(
                    t, self, X, y, sample_weight, i, len(trees),
                    verbose = self.verbose, class_weight = self.class_weight,
                    n_samples_bootstrap = n_samples_bootstrap)
                for i, t in enumerate(trees))
                             
        self.estimators_.extend(trees)
        
        if self.oob_score:
            self._set_oob_score(X, y)
        
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes = self.classes_[0]
        
        return self
        
    def _set_oob_score(self, X, y):    
       pass
   
    def _validate_y_class_weight(self, y):
       
       return y, None
   
    def _validate_X_predict(self, X):
        
        check_is_fitted(self)
        
        return self.estimators_[0].validata_X_predict(X, check_input = True)
    
    def feature_importances_(self):
        
        check_is_fitted(self)
        
        all_importances = Parallel(n_jobs = -1, 
                                   **_joblib_parallel_args(prefer = 'threads'))(
                          delayed(getattr)(tree, 'feature_importances_')
                          for tree in self.estimators_ if tree.tree_.node_count > 1)
                                       
        if not all_importances:
            return np.zeros(self.n_features_, dtype = np.float64)
        
        all_importances = np.mean(all_importances, 
                                  axis = 0, dtype = np.float64)
        
        return all_importances / np.sum(all_importances)

    
def _accumalate_prediction(predict, X, out, lock):
    
    prediction = predict(X, check_input = False)
    with lock:
        if len(out) == 1:
            out[0] += prediction
        else:
            for i in range(len(out)):
                out[i] += prediction[i]

class ForestClassifier(ClassifierMixin, BaseForest, metaclass = ABCMeta):
    
    @abstractmethod
    def __init__(self,
                 base_estimator,
                 n_estimators=100, *,
                 estimator_params=tuple(),
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None,
                 max_samples=None):
        super().__init__(
            base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            max_samples=max_samples)
    
    def predict_proba(self, X):
        
        check_is_fitted(self)
        
        X = self._validate_X_predict(X)
        
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)
        
        all_proba = [np.zeros((X.shape[0], j), dtype = np.float64)
                     for j in np.atleast_1d(self.n_classes_)]
        
        lock = threading.Lock()
        Parallel(n_jobs = -1, verbose = self.verbose,
                 **_joblib_parallel_args(require="sharedmem"))(
            delayed(accumalate_prediction)(e.predict_proba, X, all_proba, lock)
            for e in self.estimators_)
                     
        for proba in all_proba:
            proba /= len(self.estimators_)
        
        if len(all_proba) == 1:
            return all_proba[0]
        else:
            return all_proba
        
    def predict(self, X):
        
        proba = self.predict_proba(X)
        
        if self.n_outputs_ == 1:
            return self.classes_.take(np.argmax(proba, axis = 1), axis = 0)
        else:
            n_samples = proba[0].shape[0]
            class_type = self.classes_[0].dtype
            predictions = np.empty((n_samples, self.n_outputs_),
                                   dtype = class_type)
            
            for k in range(self.n_outputs_):
                predictions[:, k] = self.classes_[k].take(np.argmax(proba[k], axis = 1), axis = 0)
            
            return predictions
                     
        















































