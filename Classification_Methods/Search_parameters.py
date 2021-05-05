# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 14:17:16 2020

@author: biomet
"""

from abc import ABCMeta, abstractmethod
from collections import defaultdict
from collections.abc import Mapping, Sequence, Iterable
from functools import partial, reduce
from itertools import product
import numbers
import operator
import time
import warnings

import numpy as np
import pandas as pd
from numpy.ma import MaskedArray
from scipy.stats import rankdata
from joblib import Parallel, delayed

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.base import BaseEstimator, is_classifier, clone
from sklearn.base import MetaEstimatorMixin
from sklearn.model_selection._split import check_cv
from sklearn.model_selection._validation import _fit_and_score
from sklearn.model_selection._validation import _aggregate_score_dicts
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_random_state
from sklearn.utils.random import sample_without_replacement
from sklearn.utils.validation import indexable, check_is_fitted, _check_fit_params
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.metrics import check_scoring
from sklearn import metrics
from sklearn.utils import deprecated
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split

class ParameterGrid:
    
    def __init__(self, param_grid):
        if isinstance(param_grid, Mapping):
            param_grid = [param_grid]
            
        self.param_grid = param_grid
        
    def __iter__(self):
        
        for p in self.param_grid:
            
            items = sorted(p.items())
            if not items:
                yield {}
            else:
                keys, values = zip(*items)
                for v in product(*values):
                    params = dict(zip(keys, v))
                    yield params
                    
    def __len__(self):
        
        product = partial(reduce, operator.mul)
        return sum(product(len(v) for v in p.values()) if p else 1 
                   for p in self.param_grid)
    
    def __getitem__(self, ind):
        
        for sub_grid in self.param_grid:
            
            keys, values_lists = zip(*sorted(sub_grid.items())[::-1])
            sizes = [len(v_list) for v_list in values_lists]
            total = np.product(sizes)
            
            out = {}
            for key, v_list, n in zip(keys, values_lists, sizes):
                ind, offset = divmod(ind, n)
                out[key] = v_list[offset]
                
            return out

def fit_grid_point(X, y, estimator, parameters, train, test, 
                   scorer, verbose, error_score = np.nan, **fit_params):
    check_scoring(estimator, scorer)
    scores, n_samples_test = _fit_and_score(estimator, X, y, scorer, train, test,
                                            verbose, parameters, fit_params=fit_params,
                                            return_n_test_samples=True,
                                            error_score = error_score)
    return scores, parameters, n_samples_test

def _check_param_grid(param_grid):
    if hasattr(param_grid, 'items'):
        param_grid = [param_grid]

class BaseSearchCV(MetaEstimatorMixin, BaseEstimator, metaclass=ABCMeta):
    
    def __init__(self, estimator, *, scoring=None, n_jobs=None, iid='deprecated',
                 refit=True, cv=None, verbose=0, pre_dispath='2*n_jobs', 
                 error_score=np.nan, return_train_score=True):
        self.scoring = scoring
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.iid = iid
        self.refit = refit
        self.cv = cv
        self.verbose = verbose
        self.pre_dispath = pre_dispath
        self.error_score = error_score
        self.return_train_score = return_train_score
        
    def _run_search(self, evaluate_candidates):
        
        raise NotImplementedError("_run_search not implemented")

class ParameterRandom:
    
    def __init__(self, param_distributions, n_iter, random_state=42):
        
        if isinstance(param_distributions, Mapping):
            param_distributions = [param_distributions]
        
        self.n_iter = n_iter
        self.random_state = random_state
        self.param_distributions = param_distributions
    
    def __iter__(self):
        
        rng = check_random_state(self.random_state)
        
        for _ in range(self.n_iter):
            dist = rng.choice(self.param_distributions)
            items = sorted(dist.items())
            params = dict()
            for k, v in items:
                if hasattr(v, "rvs"):
                    params[k] = v.rvs(random_state = rng)
                else:
                    params[k] = v[rng.randint(len(v))]
            yield params
                    
###############################################################################
############################## Read data set ##################################
###############################################################################

cdi_meta = pd.read_csv("cdi_meta.csv").set_index("sample_id")
cdi_microbiome = pd.read_csv("cdi_OTUs.csv").set_index("index")

microbiome = cdi_microbiome
y = cdi_meta["DiseaseState"]
y = cdi_meta["DiseaseState"].apply(lambda x: 0 
                                          if x == "CDI" else 1
                                          if x == "ignore-nonCDI" else 2)
class_name = ["CDI", "ignore-nonCDI", "Health"]
X_train, X_test, y_train, y_test = train_test_split(microbiome, y, test_size=0.3, random_state=42)            
            
# =============================================================================
# Test Grid search
# =============================================================================

alphas = np.logspace(0,1,1000)
tuned_parameters = [{"alpha": alphas}]

grid = ParameterGrid(param_grid = tuned_parameters)
grid_loop = list(grid)

estimator = MultinomialNB()

cv = 2
cv = check_cv(cv, y, classifier = is_classifier(estimator))

X = microbiome
groups = None
X, y , groups = indexable(X,y,groups)
n_splits = cv.get_n_splits(X,y,groups)

def run_grid(X_train, y_train, X_test, y_test, param_grid):
    
    param = param_grid
    estimator = MultinomialNB(alpha = param["alpha"])
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    test_accuracy = metrics.accuracy_score(y_test, y_pred, normalize=True) * 100
    
    return param, test_accuracy

max_runs = len(grid_loop)

start = time.time()
all_param = []
all_acc = []
    
for param in grid:
    param, acc = run_grid(X_train, y_train, X_test, y_test, 
                          param)
    all_param.append(param)
    all_acc.append(acc)
time.time() - start


start = time.time()
parallel = Parallel(n_jobs = -1, verbose = 0, pre_dispatch = '2*n_jobs')
out = parallel(delayed(run_grid)(
    X_train, y_train, X_test, y_test, param)
    for param in grid)

time.time() - start

# Original version 

alphas = np.logspace(0,1,1000)
tuned_parameters = [{"alpha": alphas}]
n_folds = 10
model = MultinomialNB()
my_cv = TimeSeriesSplit(n_splits=n_folds).split(X_train)

start = time.time()
gsearch_cv = GridSearchCV(estimator = model, param_grid = tuned_parameters, 
                          cv = 2, scoring="f1_macro", n_jobs=-1)
gsearch_cv.fit(X_train, y_train)
nb_best = gsearch_cv.best_estimator_
nb_best.fit(X_train, y_train)
y_pred = nb_best.predict(X_test)
test_accuracy = metrics.accuracy_score(y_test, y_pred, normalize = True) * 100

time.time() - start

# =============================================================================
# Test Random search
# =============================================================================

# Numer of trees are used
n_estimators = [5, 10, 50, 100, 150, 200, 250, 300]

# Maximum depth of each tree
max_depth = [5, 10, 25, 50, 75, 100]
    
# Minimum number of samples per leaf 
min_samples_leaf = [1, 2, 4, 8, 10]

# Minimum number of samples to split a node
min_samples_split = [2, 4, 6, 8, 10]
    
# Maximum numeber of features to consider for making splits
max_features = ["auto", "sqrt", "log2", None]
        
criterion = ["gini", "entropy"]
    
hyperparameter = {'n_estimators': n_estimators,
                      'max_depth': max_depth,
                      'min_samples_leaf': min_samples_leaf,
                      'min_samples_split': min_samples_split,
                      'max_features': max_features, 
                      'criterion': criterion}

random = ParameterRandom(param_distributions = hyperparameter, n_iter=30, random_state = 42)
random_loop = list(random)

def run_grid(X_train, y_train, X_test, y_test, param_random):
    
    param = param_random
    estimator = RandomForestClassifier(n_estimators = param["n_estimators"],
                                       max_depth = param["max_depth"],
                                       min_samples_leaf = param["min_samples_leaf"],
                                       min_samples_split = param["min_samples_split"],
                                       max_features = param["max_features"],
                                       criterion = param["criterion"],
                                       random_state=42,
                                       n_jobs = -1)
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    test_accuracy = metrics.accuracy_score(y_test, y_pred, normalize=True) * 100
    
    return param, test_accuracy


start = time.time()
all_param = []
all_acc = []
    
for param in random_loop:
    param, acc = run_grid(X_train, y_train, X_test, y_test, 
                          param)
    all_param.append(param)
    all_acc.append(acc)
time.time() - start


start = time.time()
parallel = Parallel(n_jobs = -1, verbose = 0, pre_dispatch = '2*n_jobs')
out = parallel(delayed(run_grid)(
    X_train, y_train, X_test, y_test, param)
    for param in random_loop)

time.time() - start







































