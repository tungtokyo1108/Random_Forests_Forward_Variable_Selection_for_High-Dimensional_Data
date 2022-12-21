#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 11:19:18 2020

@author: tungbioinfo
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
from sklearn.model_selection import train_test_split
import pickle 
import os, sys
from joblib import Parallel, delayed
import PCA_Analysis as pca
import RF_Analysis_Multiclass as rfc
import RF_Analysis_Binary as rfb
from Auto_ML_Multiclass import AutoML_classification

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
class_name = ['CDI Case','Diarrheal Control', 'Non-Diarrheal Control']


X_train, X_test, y_train, y_test = train_test_split(microbiome, y, test_size=0.3, random_state=42)


# ---------------------------------------------------------------------------#

clinical_data = cdi_meta[["age", "race", "gender", "antibiotics >3mo", 
                                "antacid", "weight",
                               "Healthworker", "historyCdiff", "Surgery6mos",
                               "Vegetarian", "ResidenceCdiff"]]
clinical_data.to_csv("cdi_schubert_clinical.csv")
numeric = clinical_data["age"]
categorial = clinical_data[["race", "gender", "antibiotics >3mo", 
                                "antacid", "weight", 
                               "Healthworker", "historyCdiff", "Surgery6mos",
                               "Vegetarian", "ResidenceCdiff"]]
categorial = pd.get_dummies(categorial)
clinical_features = pd.concat([numeric, categorial], axis=1)
clinical_features = clinical_features.rename(columns={"antibiotics >3mo_no": "antibiotics_3mo_no", 
                                                      "antibiotics_3mo_yes": "antibiotics_3mo_yes",
                                   "weight_<100": "weight_100", "weight_>=250": "weight_=250"})
y = cdi_meta["DiseaseState"]
y = cdi_meta["DiseaseState"].apply(lambda x: 0 
                                          if x == "CDI" else 1
                                          if x == "ignore-nonCDI" else 2)
X_train, X_test, y_train, y_test = train_test_split(clinical_features, y, test_size=0.3, random_state=42)

class_name = ['CDI Case','Diarrheal Control', 'Non-Diarrheal Control']

###############################################################################
######################## Step 1 - Run Auto_ML #################################
###############################################################################

automl = AutoML_classification()

result = automl.fit(X_train, y_train, X_test, y_test)

###############################################################################
################### Step 2 - Run selected models ##############################
###############################################################################

dt_best, _, _, _, _ = automl.Decision_Tree(X_train, y_train, X_test, y_test)
evaluate_dt = automl.evaluate_multiclass(dt_best, X_train, y_train, X_test, y_test,
                            model = "Decision_Tree", num_class=3, class_name = class_name)

start = time.time()
st_best, _, _, _, _ = automl.Stochastic_Gradient_Descent(X_train, y_train, X_test, y_test)
end = time.time() - start
evaluate_st = automl.evaluate_multiclass(st_best, X_train, y_train, X_test, y_test,
                            model = "Stochastic_Gradient_Descent", num_class=3, class_name = class_name)

start = time.time()
st_best, _, _, _, _ = automl.Support_Vector_Classify(X_train, y_train, X_test, y_test)
end = time.time() - start
evaluate_st = automl.evaluate_multiclass(st_best, X_train, y_train, X_test, y_test,
                            model = "Support_Vector_Classify", num_class=3, class_name = class_name)

rf_best, _, _, _, _ = automl.Random_Forest(X_train, y_train, X_test, y_test)
evaluate_rf = automl.evaluate_multiclass(rf_best, X_train, y_train, X_test, y_test,
                            model = "Random Forest", num_class=2, top_features=20, class_name = class_name)


###############################################################################
################### Step 3 - Run forward algorithm ############################
###############################################################################

import itertools
from scipy import interp
from itertools import cycle
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier
from datetime import datetime as dt
import warnings 
warnings.filterwarnings("ignore")
    
st_t = dt.now()
    
n_samples, n_features = X_train.shape

n_estimators = [5, 10, 50, 100, 150, 200, 250, 300]
max_depth = [5, 10, 25, 50, 75, 100]
min_samples_leaf = [1, 2, 4, 8, 10]
min_samples_split = [2, 4, 6, 8, 10]
max_features = ["auto", "sqrt", "log2", None]

hyperparameter = {'n_estimators': n_estimators,
                  'max_depth': max_depth,
                  'min_samples_leaf': min_samples_leaf,
                  'min_samples_split': min_samples_split,
                  'max_features': max_features,
                  }

base_model_rf = RandomForestClassifier(criterion="gini", random_state=42)
n_iter_search = 30
scoring = "accuracy"
n_selected_features = 1500

# selected feature set, initialized to be empty
F = []
count = 0
ddict = {}
all_F = []
all_c = []
all_acc = []
all_model = []
start = time.time()
while count < n_selected_features:
    max_acc = 0
    time_loop = time.time()
    
    for i in X_train.columns:
            if i not in F:
                F.append(i)
                X_train_tmp = X_train[F]
                acc = 0
                rsearch_cv = RandomizedSearchCV(estimator=base_model_rf,
                                                random_state=42,
                                                param_distributions=hyperparameter,
                                                n_iter=n_iter_search,
                                                #cv=cv_timeSeries,
                                                cv=2,
                                                scoring=scoring,
                                                n_jobs=-1)
                rsearch_cv.fit(X_train_tmp, y_train)
                best_estimator = rsearch_cv.best_estimator_
                y_pred = best_estimator.predict(X_test[F])
                acc = metrics.accuracy_score(y_test, y_pred)
                F.pop()
                if acc > max_acc:
                    max_acc = acc
                    idx = i
                    best_model = best_estimator
                    
    F.append(idx)
    count += 1
        
    print("The current number of features: {} - Accuracy: {}%".format(count, round(max_acc*100, 2)))
    print("Time for computation: {}".format(time.time() - time_loop))

    all_F.append(np.array(F))
    all_c.append(count)
    all_acc.append(max_acc)
    all_model.append(best_model)
    
time.time() - start    
    
c = pd.DataFrame(all_c)
a = pd.DataFrame(all_acc)
f = pd.DataFrame(all_F)    
f["All"] = f[f.columns[0:]].apply(
        lambda x: ', '.join(x.dropna().astype(str)), axis=1)
    
all_info = pd.concat([c, a, f["All"]], axis=1)    
all_info.columns = ['Num_feature', 'Accuracy', 'Feature']    
all_info = all_info.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)

# =============================================================================
# Test Parallel algorithm
# =============================================================================
    
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from collections.abc import Mapping, Sequence, Iterable
from functools import partial, reduceChanglin
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

from sklearn.utils.fixes import _joblib_parallel_args
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.model_selection import train_test_split

###################################################################################

def run_parallel(estimator, X_train, y_train, X_test, y_test, i, F, max_acc, idx, y_pred, best_estimator):
    
    if i not in F:
        #idx = i
        #y_pred = y_test
        F.append(i)
        X_train_tmp = X_train[F]
        acc = 0
        rsearch_cv = RandomizedSearchCV(estimator=estimator,
                                                random_state=42,
                                                param_distributions=hyperparameter,
                                                n_iter=n_iter_search,
                                                #cv=cv_timeSeries,
                                                cv=2,
                                                scoring=scoring,
                                                n_jobs=-1)
        rsearch_cv.fit(X_train_tmp, y_train)
        best_estimator = rsearch_cv.best_estimator_
        y_pred = best_estimator.predict(X_test[F])
        acc = metrics.accuracy_score(y_test, y_pred)
        F.pop()
        if acc > max_acc:
            max_acc = acc
            idx = i
            best_model = best_estimator
    
    return idx, best_estimator, max_acc

# selected feature set, initialized to be empty
F = []
count = 0
ddict = {}
all_F = []
all_c = []
all_acc = []
all_model = []
start = time.time()
# Restart algorithm
# F.pop(-1)
while count < n_selected_features:
    max_acc = 0
    time_loop = time.time()
    
    parallel = Parallel(n_jobs = -1, verbose = 1, pre_dispatch = '2*n_jobs', prefer="processes")
    #parallel = Parallel(n_jobs = -1, verbose = 1, **_joblib_parallel_args(prefer='threads'))
    out = parallel(delayed(run_parallel)(
        base_model_rf, X_train, y_train, X_test, y_test, i, F, max_acc, i, y_test, base_model_rf)
        for i in X_train.columns)
    
    idx_list = []
    best_model_list = []
    max_acc_list = []
    for i in range(0, len(out)):
        idx_list.append(out[i][0])
        best_model_list.append(out[i][1])
        max_acc_list.append(out[i][2])
    
    idx_list = pd.Series(idx_list)
    best_model_list = pd.Series(best_model_list)
    max_acc_list = pd.Series(max_acc_list)
    
    total = pd.concat([idx_list, best_model_list, max_acc_list], axis=1)   
    total.columns = ['Feature', 'Model','Accuracy']    
    total = total.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
                    
    F.append(total.iloc[0]["Feature"])
    count += 1
        
    print("The current number of features: {} - Accuracy: {}%".format(count, round(total.iloc[0]["Accuracy"]*100, 2)))
    print("Time for computation: {}".format(time.time() - time_loop))
    
    # Resart forward algorithm
    #count = len(F) - 1
    #count += 1
    #print("The current number of features: {} - Accuracy: {}%".format(count, round(total.iloc[0]["Accuracy"]*100, 2)))
    #print("Time for computation: {}".format(time.time() - time_loop))

    all_F.append(np.array(F))
    all_c.append(count)
    all_acc.append(total.iloc[0]["Accuracy"])
    all_model.append(total.iloc[0]["Model"])

time.time() - start

c = pd.DataFrame(all_c)
a = pd.DataFrame(all_acc)
f = pd.DataFrame(all_F)    
f["All"] = f[f.columns[0:]].apply(
        lambda x: ', '.join(x.dropna().astype(str)), axis=1)
    
all_info = pd.concat([c, a, f["All"]], axis=1)    
all_info.columns = ['Num_feature', 'Accuracy', 'Feature']    
all_info = all_info.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)




















