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
cdi_function = pd.read_csv("cdi_OTUs_functions_100_120.csv", index_col=0)
cdi_gmm = pd.read_csv("CDI_gmm_med_pred.csv", index_col=0)
cdi_gmm = cdi_gmm.loc[:, (cdi_gmm != 0).any(axis=0)]

internal_node = pd.read_csv("Internal_node.csv", index_col=0)
#internal_node = internal_node.set_index("Unnamed: 0")

microbiome = cdi_microbiome

microbiome_internal = pd.concat([microbiome, internal_node], axis=1)

y = cdi_meta["DiseaseState"]
y = cdi_meta["DiseaseState"].apply(lambda x: 0 
                                          if x == "CDI" else 1
                                          if x == "ignore-nonCDI" else 2)
class_name = ['CDI Case','Diarrheal Control', 'Non-Diarrheal Control']


crc_meta = pd.read_csv("CRC_meta_data.csv").set_index("Sample_Name_s")
crc_microbiome = pd.read_csv("CRC_OTU_data.csv").set_index("Sample_Name_s")
crc_microbiome = crc_microbiome.divide(crc_microbiome.sum(axis=1), axis=0)
y = crc_meta["DiseaseState"]
y = crc_meta["DiseaseState"].apply(lambda x: 0 
                                          if x == "CRC" else 1)
class_name = ['CRC Case','Healthy Control']

tottori_2019_raw = pd.read_csv("2019_Tottori_Microbiomics.csv", index_col="Sample_ID")
tottori_2019_df = tottori_2019_raw.drop(columns = ['level_0', 'index', 'Sample_ID.1'])
tottori_2019_df = tottori_2019_df.divide(tottori_2019_df.sum(axis=1), axis=0)
y = tottori_2019_raw["level_0"].apply(lambda x: 0 
                                      if x == "W1" else 1)
class_name = ['W1','W4']

tottori_2019_metabolo = pd.read_csv("2019_Tottori_Metabolomics.csv", index_col="plot")
tottori_2019_meta_df = tottori_2019_metabolo.drop(columns = ['index', 'plot.1'])
y = tottori_2019_metabolo["index"].apply(lambda x: 0 
                                      if x == "W1" else 1)
class_name = ['W1','W4']


X_train, X_test, y_train, y_test = train_test_split(microbiome, y, test_size=0.3, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(internal_node, y, test_size=0.3, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(microbiome_internal, y, test_size=0.3, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(cdi_function, y, test_size=0.3, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(cdi_gmm, y, test_size=0.3, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(tottori_2019_df, y, test_size=0.3, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(tottori_2019_meta_df, y, test_size=0.3, random_state=42)

# CRC 16S microbiome 
X_train, X_test, y_train, y_test = train_test_split(crc_microbiome, y, test_size=0.3, random_state=42)


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

result.to_csv("Internal_node_model_testing.csv")

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

start = time.time()
xgb_best, _, _, _, _ = automl.Gradient_Boosting(X_train, y_train, X_test, y_test)
end = time.time() - start
evaluate_xgb = automl.evaluate_multiclass(xgb_best, X_train, y_train, X_test, y_test,
                            model = "XGBoost", num_class=3, top_features=20, class_name = class_name)

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

all_info.to_csv("CDI_subset_accuracy.csv", index=False)
f.to_csv("CDI_subset.csv")
with open("CDI_models.txt", "wb") as fp:
    pickle.dump(all_model, fp)

# =============================================================================
# Forward algorithm + Extreme Gradient Boosting
# =============================================================================

n_samples, n_features = X_train.shape

n_estimators = [5, 10, 50, 100, 150, 200, 250, 300]
max_depth = [5, 10, 25, 50, 75, 100]
min_child_weight = [5, 10, 25, 50, 75, 100]
gamma = [0.5, 1, 1.5, 2, 5]
subsample = [0.2, 0.4, 0.6, 0.8, 1]
colsample_bytree = [0.2, 0.4, 0.6, 0.8, 1]
hyperparameter = {'n_estimators': n_estimators,
                      'max_depth': max_depth,
                      'min_child_weight': min_child_weight,
                      'gamma': gamma,
                      'subsample': subsample,
                      'colsample_bytree': colsample_bytree}

xgb = XGBClassifier(learning_rate=0.02, objective='multi:softmax', silent=True, nthread=-1)
n_iter_search = 30
scoring = "accuracy"
n_selected_features = 100

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
                rsearch_cv = RandomizedSearchCV(estimator = xgb,
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

all_info.to_csv("CDI_gmm_med_subset_accuracy.csv", index=False)
f.to_csv("CDI_gmm_med_subset.csv")
with open("CDI_gmm_med_models.txt", "wb") as fp:
    pickle.dump(all_model, fp)


# =============================================================================
# Test Parallel algorithm
# =============================================================================
    
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
        
criterion = ["gini"]
    
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
                                       random_state=42)
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    test_accuracy = metrics.accuracy_score(y_test, y_pred, normalize=True) * 100
    
    return param, test_accuracy

start = time.time()
parallel = Parallel(n_jobs = -1, verbose = 10, pre_dispatch = '2*n_jobs')
out = parallel(delayed(run_grid)(
    X_train, y_train, X_test, y_test, param)
    for param in random_loop)

time.time() - start

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

all_info.to_csv("CRC_subset_16S_accuracy.csv", index=False)
f.to_csv("CRC_subset_16S.csv")
with open("CRC_subset_16S_models.txt", "wb") as fp:
    pickle.dump(all_model, fp)

# =============================================================================
# Test Parallel_RF_FVS 
# =============================================================================

import RF_Analysis_Multiclass as rfc
from RF_Analysis_Multiclass import Parallel_RF_FVS
import RF_Analysis_Binary as rfb
from Auto_ML_Multiclass import AutoML_classification

all_info, all_model, f = rfc.RF_FVS(X_train, y_train, X_test, y_test, n_selected_features = 1000, scoring='accuracy')

all_info, f, all_model = rfc.Parallel_simple_RF_FVS(X_train, y_train, X_test, y_test)

par_rf_fvs = Parallel_RF_FVS()
all_info, f, all_model = par_rf_fvs.RF_FVS(X_train, y_train, X_test, y_test)

# =============================================================================
# Test accuracy model 
# =============================================================================

all_features_grid = pd.read_csv("CDI_Boruta_1008_subset.csv")
all_info_grid = pd.read_csv("CDI_Boruta_1008_subset_accuracy.csv")
with open("CDI_Boruta_1008_models.txt", "rb") as fp:
    load_grid_model = pickle.load(fp)
subset = all_features_grid.drop(columns = ["Unnamed: 0", "All"])

best_model_95 = load_grid_model[95]
subset = subset.iloc[95].dropna()
microbiome_subset = cdi_microbiome[subset]
microbiome_subset.to_csv("CDI_Selected_11_Metabolic.csv")

X_train, X_test, y_train, y_test = train_test_split(microbiome_subset, y, test_size=0.3, random_state=42)

evaluate_rf = automl.evaluate_multiclass(best_model_95, X_train, y_train, X_test, y_test,
                            model = "Random Forest", num_class=2, top_features=90, class_name = class_name)

# =============================================================================
# Test function inference
# =============================================================================

function_name = pd.read_csv("cdi_schubert_subset.csv", index_col=0)
function_name = function_name.drop(columns=["Full_name"])
function_name = function_name.T
#function_name = microbiome_subset
cdi_group = cdi_meta["DiseaseState"]
cdi_group = pd.get_dummies(cdi_group)
drug_group = cdi_meta[["antibiotics >3mo", "protonpump"]]
drug_group = drug_group.rename(columns= {'antibiotics >3mo': 'antibiotics'})
drug_group = pd.get_dummies(drug_group)
cdi_group = cdi_group.rename(columns={'CDI': 'CDI Case', 'ignore-nonCDI': 'Diarrheal Control',
                                      'H': 'Non-Diarrheal Control'})
corr_function_cdi = pd.concat([cdi_group, drug_group, function_name], axis=1)
#corr_function_cdi = pd.concat([cdi_group, function_name], axis=1)

plt.subplots(figsize=(10,40))
#corr = corr_function_cdi.corr()
corr = corr_function_cdi.corr(method = "spearman")
corr = corr.drop(index=['CDI Case', 'Non-Diarrheal Control', 'Diarrheal Control', 
                        'antibiotics_no', 'antibiotics_yes', 'protonpump_no', 'protonpump_yes'])
corr = corr[['CDI Case','antibiotics_yes', 'protonpump_yes', 'Non-Diarrheal Control']]
#corr = corr.drop(index=['CDI Case', 'Non-Diarrheal Control', 'Diarrheal Control'])
#corr = corr[['CDI Case','Non-Diarrheal Control', 'Diarrheal Control']]
cmap = sns.diverging_palette(220, 10, as_cmap=True)
hm = sns.heatmap(round(corr,2), annot=True, cmap=cmap, fmt=".2f",annot_kws={"size": 20},
                 linewidths=.05)
hm.set_xticklabels(hm.get_xticklabels(), fontsize = 20, rotation=45, horizontalalignment='right')
hm.set_yticklabels(hm.get_yticklabels(), rotation=0, fontsize = 20)

otu_cdi = corr.loc[(corr['CDI Case'] > 0.1) & (corr["Non-Diarrheal Control"] < 0)]
otu_cdi = otu_cdi.drop(columns=["Non-Diarrheal Control"])
otu_health = corr.loc[(corr['CDI Case'] < 0) & (corr["Non-Diarrheal Control"] > 0.1)]
otu_health = otu_health.drop(columns=["CDI Case"])
otu_cdi.to_csv("otu_cdi.csv")
otu_health.to_csv("otu_health.csv")

# =============================================================================
# Test metabolic inference
# =============================================================================

metabolic_name = pd.read_csv("CDI_Selected_11_Metabolic_name.csv", index_col=0)
metabolic_name = metabolic_name.loc[:, (metabolic_name != 0).any(axis=0)]
cdi_group = cdi_meta["DiseaseState"]
cdi_group = pd.get_dummies(cdi_group)
cdi_group = cdi_group.rename(columns={'CDI': 'CDI Case', 'ignore-nonCDI': 'Diarrheal Control',
                                      'H': 'Non-Diarrheal Control'})
corr_metabolic_cdi = pd.concat([cdi_group, metabolic_name], axis=1)

plt.subplots(figsize=(10,20))
corr = corr_metabolic_cdi.corr(method = "spearman")
corr = corr.drop(index=['CDI Case', 'Non-Diarrheal Control', 'Diarrheal Control'])
corr = corr[['CDI Case','Non-Diarrheal Control']]
cmap = sns.diverging_palette(220, 10, as_cmap=True)
hm = sns.heatmap(round(corr,2), annot=True, cmap=cmap, fmt=".2f",annot_kws={"size": 20},
                 linewidths=.05)
hm.set_xticklabels(hm.get_xticklabels(), fontsize = 20, rotation=45, horizontalalignment='right')
hm.set_yticklabels(hm.get_yticklabels(), rotation=0, fontsize = 20)

# =============================================================================
# Test variable importance
# =============================================================================

impt = evaluate_rf["importance"].set_index("Features")

microbiome_species = list(impt.index)
microbiome_species_org = microbiome_species
microbiome_species = [i.split(';')[-1][3:] for i in microbiome_species]
microbiome_species = pd.DataFrame(microbiome_species, columns = ["Species"]).reset_index(drop=True)
microbiome_species_org = pd.DataFrame(microbiome_species_org, columns = ["Full_name"]).reset_index(drop=True)

microbiome_f_br = [i.split(';')[-4][3:] for i in microbiome_species]
microbiome_f_br = pd.DataFrame(microbiome_f_br, columns = ["Family"]).reset_index(drop=True)

microbiome_species_br = pd.concat([microbiome_f_br, microbiome_species, microbiome_species_org], axis=1)
microbiome_species_br.to_csv("F_S_BR.csv")

impt_speies = pd.concat([microbiome_species, evaluate_rf["importance"]], axis=1)
impt_speies = impt_speies.drop(columns = ["Features"])
impt_speies_200 = impt_speies.iloc[0:200]
impt_speies_200 = impt_speies_200.set_index("Species")
species_200 = cdi_microbiome[impt_speies_200.index]
impt_speies.to_csv("Imp_var_3300.csv", index=False, sep='\t')
#impt_speies = impt_speies.sort_values(by="Species", ascending=True)

index = impt_speies["Species"].iloc[100:200]
importance_desc = impt_speies["Importance"].iloc[100:200]
feature_space = []
#for i in range(indices.shape[0]-1, -1, -1):
#    feature_space.append(X_train.columns[indices[i]])
    
fig, ax = plt.subplots(figsize=(20,20))
ax = plt.gca()
#plt.title("Feature importances", fontsize=30)
plt.barh(index, importance_desc, align="center", color="blue", alpha=0.6)
plt.grid(axis="x", color="white", linestyle="-")
plt.xlabel("The average of decrease in impurity", fontsize=20)
plt.ylabel("Features", fontsize=20)
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)
plt.show()

# =============================================================================
# Validate the performance of approaches 
# =============================================================================

## Accuracy

rf_best.fit(X_train, y_train)
y_pred = rf_best.predict(X_test)
y_pred_prob = rf_best.predict_proba(X_test)
#y_pred_prob = rf_best.predict(X_test)  
y_test_cat = np.array(pd.get_dummies(y_test))

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(3):
    fpr[i], tpr[i], _ = metrics.roc_curve(y_test_cat[:,i], y_pred_prob[:,i])
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])
  
test_accuracy_RF = accuracy_score(y_test, y_pred, normalize=True) * 100
test_accuracy_RF_PI = accuracy_score(y_test, y_pred, normalize=True) * 100
test_accuracy_RF_BR = accuracy_score(y_test, y_pred, normalize=True) * 100
test_accuracy_RF_FVS = accuracy_score(y_test, y_pred, normalize=True) * 100

accuracy_RF = {'Methods': ['RF', 'RF_PI', 'RF_BR', 'RF_FVS'],
               'Accuracy': [test_accuracy_RF, test_accuracy_RF_PI, test_accuracy_RF_BR, test_accuracy_RF_FVS]}
accuracy_RF_df = pd.DataFrame.from_dict(accuracy_RF)
accuracy_RF_df.to_csv("Accuracy_RFs_CDI.csv", index=False, sep='\t')

fig, ax = plt.subplots(figsize=(10,10))
ax = plt.gca()
#plt.title("Feature importances", fontsize=30)
plt.barh(accuracy_RF_df["Methods"], accuracy_RF_df["Accuracy"], align="center", color="blue", alpha=0.6)
plt.grid(axis="x", color="white", linestyle="-")
plt.xlabel("Accuracy", fontsize=20)
plt.ylabel("Methods", fontsize=20)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.show()

fpr_RF = fpr[0]
tpr_RF = tpr[0]
roc_auc_RF = roc_auc[0]

fpr_RF_PI = fpr[0]
tpr_RF_PI = tpr[0]
roc_auc_RF_PI = roc_auc[0]

fpr_RF_BR = fpr[2]
tpr_RF_BR = tpr[2]
roc_auc_RF_BR = roc_auc[2]

fpr_RF_FVS = fpr[2]
tpr_RF_FVS = tpr[2]
roc_auc_RF_FVS = roc_auc[2]

d_fpr = {'RF': fpr_RF, 'RF_PI': fpr_RF_PI, 'RF_BR': fpr_RF_BR, 'RF_FVS': fpr_RF_FVS}
with open("CDI_fpr.txt", "wb") as fp:
    pickle.dump(d_fpr, fp)
    
d_tpr = {'RF': tpr_RF, 'RF_PI': tpr_RF_PI, 'RF_BR': tpr_RF_BR, 'RF_FVS': tpr_RF_FVS}
with open("CDI_tpr.txt", "wb") as fp:
    pickle.dump(d_tpr, fp)
    
d_roc_auc = {'RF': roc_auc_RF, 'RF_PI': roc_auc_RF_PI, 'RF_BR': roc_auc_RF_BR, 'RF_FVS': roc_auc_RF_FVS}
with open("CDI_auc.txt", "wb") as fp:
    pickle.dump(d_roc_auc, fp)


methods = ['RF', 'RF_PI', 'RF_BR', 'RF_FVS']
fig, ax = plt.subplots(figsize=(12,12))
ax = plt.gca()
colors = sns.color_palette()
for i, color in zip(methods, colors):
    plt.plot(d_fpr[i], d_tpr[i], color=color, lw=2,
        label = "{0} (AUC = {1:0.2f})".format(i, d_roc_auc[i]))   
plt.plot([0,1], [0,1], "k--", lw=3, color='red')
plt.xlabel("False Positive Rate", fontsize=20)
plt.ylabel("True Positive Rate", fontsize=20)
plt.legend(loc="lower right", fontsize=15)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.show()


with open("CDI_fpr.txt", "rb") as fp:
    load_fpr = pickle.load(fp)


















