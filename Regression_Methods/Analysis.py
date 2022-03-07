#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 10:16:42 2021

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
from sklearn.utils import shuffle
import pickle 
import os, sys
from joblib import Parallel, delayed
from Auto_ML_Multiclass import AutoML_classification

###############################################################################
############################## Read data set ##################################
###############################################################################

rumi = pd.read_csv("rumi.csv")
rumi = rumi.drop(rumi[rumi["Depressiongroup"]==1].index, axis=0).reset_index(drop=True)
depre_gr = rumi["Depressiongroup"].apply(lambda x: "BPD" 
                                          if x == 2 else "H"
                                          if x == 0 else "MDD")
sex = rumi["Gender_1_male"].apply(lambda x: 0 if x == 2 else 1)
rumi = rumi.drop(columns = ["Depressiongroup", "Gender_1_male"])
rumi = pd.concat([depre_gr, sex, rumi], axis = 1)
rumi = shuffle(rumi).reset_index(drop=True)


rumi_meta = rumi[['MRI_expID', 'MRI_ordID', 'CurrentDepression', 'Depressiongroup', 'TIV',
       'Age', 'Gender_1_male', 'BDI_Total', 'RRS_Brooding', 'RRS_Reflection', 'RRS_DepressiveRumination',
       'RRS_Total', 'Dep_PastEpisodes', 'Dep_Duration']]
rumi_meta = rumi_meta.set_index('MRI_expID')
sns.pairplot(rumi_meta[['RRS_Brooding', 'RRS_Reflection', 'RRS_DepressiveRumination', 'RRS_Total', 'Depressiongroup']], 
             hue="Depressiongroup")

rumi_meta_bdp = rumi_meta.loc[rumi_meta['Depressiongroup'] == "BPD"]
rumi_meta_mdd = rumi_meta.loc[rumi_meta['Depressiongroup'] == 'MDD']
sns.pairplot(rumi_meta_bdp[['RRS_Brooding', 'RRS_Reflection', 'RRS_DepressiveRumination', 'RRS_Total', 'CurrentDepression']], 
             hue="CurrentDepression")
sns.pairplot(rumi_meta_mdd[['RRS_Brooding', 'RRS_Reflection', 'RRS_DepressiveRumination', 'RRS_Total', 'CurrentDepression']], 
             hue="CurrentDepression")

rumi_region = rumi.drop(columns = ['MRI_ordID', 'CurrentDepression', 'Depressiongroup', 'TIV',
       'Age', 'Gender_1_male', 'BDI_Total', 'RRS_Brooding', 'RRS_Reflection', 'RRS_DepressiveRumination',
       'RRS_Total', 'Dep_PastEpisodes', 'Dep_Duration'])
rumi_region = rumi_region.set_index('MRI_expID')

rumi_region_T = rumi_region.T

rumi_region_bdp = rumi_region.loc[rumi_meta_bdp.index]
rumi_region_mdd = rumi_region.loc[rumi_meta_mdd.index]

y = rumi_meta["Depressiongroup"].apply(lambda x: 0 
                                          if x == "MDD" else 1
                                          if x == "BPD" else 2)

class_name = ["MDD", "BPD", 'Healthy']


y = rumi_meta["Depressiongroup"].apply(lambda x: 0 
                                          if x == "H" else 1)
class_name = ["MDD-BPD", 'Healthy']

X_train, X_test, y_train, y_test = train_test_split(rumi_region, y, test_size=0.3, random_state=42)

###############################################################################
######################## Step 1 - Run Auto_ML #################################
###############################################################################

automl = AutoML_classification()

result = automl.fit(X_train, y_train, X_test, y_test)

###############################################################################
################### Step 2 - Run selected models ##############################
###############################################################################

log_best, _, _, _, _ = automl.Decision_Tree(X_train, y_train, X_test, y_test)
evaluate_dt = automl.evaluate_multiclass(log_best, X_train, y_train, X_test, y_test,
                            model = "Logistics_regression", num_class=3, class_name = class_name)

sgd_best, _, _, _, _ = automl.Extreme_Gradient_Boosting(X_train, y_train, X_test, y_test)
evaluate_dt = automl.evaluate_multiclass(sgd_best, X_train, y_train, X_test, y_test,
                            model = "Stochastic_Gradient_Descent", num_class=3, class_name = class_name)

rf_best, _, _, _, _ = automl.Random_Forest(X_train, y_train, X_test, y_test)
evaluate_rf = automl.evaluate_multiclass(rf_best, X_train, y_train, X_test, y_test,
                            model = "Random Forest", num_class=3, top_features=20, class_name = class_name)

###############################################################################
########## Step 3.1 - Run forward algorithm + Random Forest ###################
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
from sklearn.model_selection import KFold, RepeatedStratifiedKFold, RepeatedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
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
    
base_model_rf = RandomForestClassifier(criterion = "gini", random_state=42)
n_iter_search = 30
scoring = "accuracy"
n_selected_features = 240

max_depth = [5, 10, 25, 50, 75, 100]
min_samples_leaf = [1, 2, 4, 8, 10]
min_samples_split = [2, 4, 6, 8, 10]
max_features = ["auto", "sqrt", "log2", None]
criterion = ["gini", "entropy"]
splitter = ["best", "random"]

hyperparameter = {"max_depth": max_depth,
                          "min_samples_leaf": min_samples_leaf,
                          "min_samples_split": min_samples_split,
                          "max_features": max_features,
                          "criterion": criterion,
                          "splitter": splitter}
base_model_rf = DecisionTreeClassifier(random_state = 42)
n_iter_search = 30
scoring = "accuracy"
n_selected_features = 240
    
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
                                                cv=5,
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

###############################################################################

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

all_info.to_csv("DT_subset_accuracy_multi.csv", index=False)
f.to_csv("DT_subset_multi.csv")
with open("DT_models_multi.txt", "wb") as fp:
    pickle.dump(all_model, fp)

######

all_features_grid = pd.read_csv("RF_subset_multi.csv")
all_info_grid = pd.read_csv("RF_subset_accuracy_multi.csv")
with open("RF_models_multi.txt", "rb") as fp:
    load_grid_model = pickle.load(fp)
subset = all_features_grid.drop(columns = ["Unnamed: 0"])

best_model_95 = load_grid_model[9]
subset = subset.iloc[9].dropna()
rumi_region_subset = rumi_region[subset]
rumi_region_subset.to_csv("RF_subset_multi_37.csv")

X_train, X_test, y_train, y_test = train_test_split(rumi_region_subset, y, test_size=0.3, random_state=42)

evaluate_rf = automl.evaluate_multiclass(best_model_95, X_train, y_train, X_test, y_test,
                            model = "Decision Tree", num_class=2, top_features=10, class_name = class_name)

###############################################################################
################# Step 3.1 - Run forward algorithm + SGD ######################
###############################################################################
    
from sklearn.linear_model import SGDClassifier
st_t = dt.now()
    
n_samples, n_features = X_train.shape

# Loss function 
loss = ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"]
        
penalty = ["l2", "l1", "elasticnet"]
        
# The higher the value, the stronger the regularization 
alpha = np.logspace(-7, -1, 100)
        
# The Elastic Net mixing parameter 
l1_ratio = np.linspace(0, 1, 100)
        
epsilon = np.logspace(-5, -1, 100)
        
learning_rate = ["constant", "optimal", "invscaling", "adaptive"]
        
eta0 = np.logspace(-7, -1, 100)
        
hyperparameter = {"loss": loss,
                  "penalty": penalty,
                  "alpha": alpha,
                  "l1_ratio": l1_ratio,
                  "epsilon": epsilon,
                  "learning_rate": learning_rate,
                  "eta0": eta0}

base_model_rf = SGDClassifier(n_jobs = -1)
n_iter_search = 30
scoring = "accuracy"
n_selected_features = 240
    
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
                rsearch_cv = RandomizedSearchCV(estimator = base_model_rf, 
                                                param_distributions = hyperparameter, 
                                                cv = 2,
                                                scoring = scoring, 
                                                n_iter = n_iter_search, 
                                                n_jobs = -1)
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

c = pd.DataFrame(all_c)
a = pd.DataFrame(all_acc)
f = pd.DataFrame(all_F)    
f["All"] = f[f.columns[0:]].apply(
        lambda x: ', '.join(x.dropna().astype(str)), axis=1)


###############################################################################
######## Step 4.1 - Run forward algorithm + Random_Forest_regression ##########
###############################################################################

from Auto_ML_Regression import AutoML_Regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_percentage_error
import math

y = rumi_meta["RRS_Brooding"]

rumi_region_plus = pd.concat([rumi_meta[['CurrentDepression', 'TIV', 'Age','Gender_1_male']], 
                              rumi_region], axis=1)

#-------
y = rumi_meta_bdp["BDI_Total"]
rumi_region_bdp_plus = pd.concat([rumi_meta_bdp[['CurrentDepression', 'TIV', 'Age','Gender_1_male']], 
                              rumi_region_bdp], axis=1)
X_train, X_test, y_train, y_test = train_test_split(rumi_region_bdp_plus, y, test_size=0.3, random_state=42)

# ------
y = rumi_meta_bdp["BDI_Total"]
rumi_region_mdd_plus = pd.concat([rumi_meta_mdd[['CurrentDepression', 'TIV', 'Age','Gender_1_male']], 
                              rumi_region_mdd], axis=1)
X_train, X_test, y_train, y_test = train_test_split(rumi_region_mdd_plus, y, test_size=0.3, random_state=42)

# ------
ress_BPD_brain = pd.read_csv("BPD_brain.csv", header=None)
ress_BPD_brain.columns = rumi_region.columns
ress_BPD_meta = pd.read_csv("BPD_rrs.csv", header=None)
ress_BPD_meta.columns = ['RRS_Brooding', 'RRS_Reflection', 'RRS_DepressiveRumination','RRS_Total']
y = ress_BPD_meta["RRS_Brooding"]

X_train, X_test, y_train, y_test = train_test_split(ress_BPD_brain, y, test_size=0.3, random_state=42)

# ------
ress_MDD_brain = pd.read_csv("MDD_brain.csv", header=None)
ress_MDD_brain.columns = rumi_region.columns
ress_MDD_meta = pd.read_csv("MDD_rrs.csv", header=None)
ress_MDD_meta.columns = ['RRS_Brooding', 'RRS_Reflection', 'RRS_DepressiveRumination','RRS_Total']
y = ress_MDD_meta["RRS_Brooding"]

X_train, X_test, y_train, y_test = train_test_split(ress_MDD_brain, y, test_size=0.3, random_state=42)

# ------
ress_HC_brain = pd.read_csv("Health_brain.csv", header=None)
ress_HC_brain.columns = rumi_region.columns
ress_HC_meta = pd.read_csv("Health_rrs.csv", header=None)
ress_HC_meta.columns = ['RRS_Brooding', 'RRS_Reflection', 'RRS_DepressiveRumination','RRS_Total']
y = ress_HC_meta["RRS_Brooding"]

X_train, X_test, y_train, y_test = train_test_split(ress_HC_brain, y, test_size=0.3, random_state=42)

# -------

ress_all_brain = pd.concat([ress_BPD_brain, ress_MDD_brain, ress_HC_brain]).reset_index(drop=True)
ress_all_meta = pd.concat([ress_BPD_meta, ress_MDD_meta, ress_HC_meta]).reset_index(drop=True)
y = ress_all_meta["RRS_Brooding"]

X_train, X_test, y_train, y_test = train_test_split(ress_all_brain, y, test_size=0.3, random_state=42)

automl = AutoML_Regression()

result = automl.fit(X_train, y_train, X_test, y_test)
result.to_csv("AutoML_RRS_total_rumi_region_plus.csv", index = False)

ress_BPD_meta["Label"] = "BPD"
ress_MDD_meta["Label"] = "MDD"
ress_HC_meta["Label"] = "HC"
ress = pd.concat([ress_BPD_meta, ress_MDD_meta, ress_HC_meta]).reset_index(drop=True)
sns.pairplot(ress, hue="Label")

#------------------------------------------------------------------------------
from scipy.stats import spearmanr
import seaborn as sns

automl = AutoML_Regression()

lasso_best, _, _, _ = automl.XGBoost(X_train, y_train, X_test, y_test)
lasso_best.fit(X_train, y_train)
y_pred = lasso_best.predict(X_test)

dt = {"True RRS_Brooding": y_test, "Predicted RRS_Brooding": y_pred}
df = pd.DataFrame(dt)
g = sns.lmplot(x="True RRS_Brooding", y="Predicted RRS_Brooding", data=df)
g.set(ylim = (min(y_test), max(y_test)))
g.set(xlim = (min(y_test), max(y_test)))
plt.text(-3.9, max(y_test) - 1, r'MSE = %.2f' % (mean_squared_error(y_test, y_pred)))
plt.text(-3.9, max(y_test) - 2, r'Corr = %.2f' % (spearmanr(y_test, y_pred)[0]))

plt.scatter(y_pred, y_test, s=8)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--k')
plt.xlim(min(y_test), max(y_test))
plt.ylim(min(y_test), max(y_test))
plt.ylabel('True RRS_Brooding')
plt.xlabel('Predicted RRS_Brooding')
#plt.text(s='Random Forest without Forward varible', x=1,
#            y=2, fontsize=12, multialignment='center')
#plt.text(-4.5, max(y_test) - 1, r'Corr = %.2f' % (spearmanr(y_test, y_pred)[0]))
plt.text(-4, max(y_test) - 1, r'MSE = %.2f' % (mean_squared_error(y_test, y_pred)))
#plt.text(min(y_pred), max(y_test) - 15, r'Accuracy = %.2f %' % (100 - 100*mean_absolute_percentage_error(y_test, y_pred)))
#plt.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))

errors = abs(y_pred - y_test)
mean_err = np.stack(errors/y_test)
mean_err = mean_err[np.isfinite(mean_err)]
mape = 100 * np.mean(mean_err)
acc = 100 - mape

#------------------------------------------------------------------------------

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


my_cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
base_model_rf = RandomForestRegressor(criterion="mse", random_state=42)
n_iter_search = 30

scoring = "neg_mean_squared_error"
n_selected_features = 240

F = []
count = 0
ddict = {}
all_F = []
all_c = []
all_acc = []
all_mse = []
all_model = []
start = time.time()
while count < n_selected_features:
    max_acc = 0
    min_err = np.inf
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
                                                #cv=my_cv,
                                                cv=5,
                                                scoring=scoring,
                                                n_jobs=-1)
                rsearch_cv.fit(X_train_tmp, y_train)
                best_estimator = rsearch_cv.best_estimator_
                y_pred = best_estimator.predict(X_test[F])
                mse = mean_squared_error(y_test, y_pred)
                #acc = metrics.accuracy_score(y_test, y_pred)
                F.pop()
                if mse < min_err:
                    min_err = mse
                    idx = i
                    best_model = best_estimator
                    #errors = abs(y_pred - y_test)
                    #mean_err = np.stack(errors/y_test)
                    #mean_err = mean_err[np.isfinite(mean_err)]
                    mape = mean_absolute_percentage_error(y_test, y_pred)
                    max_acc = 100 - (100*mape)
                    
    F.append(idx)
    count += 1
        
    print("The current number of features: {} - MSE: {}".format(count, round(min_err, 2)))
    print("Time for computation: {}".format(time.time() - time_loop))

    all_F.append(np.array(F))
    all_c.append(count)
    all_acc.append(max_acc)
    all_model.append(best_model)
    all_mse.append(min_err)

c = pd.DataFrame(all_c)
a = pd.DataFrame(all_acc)
f = pd.DataFrame(all_F)  
e = pd.DataFrame(all_mse)  
f["All"] = f[f.columns[0:]].apply(
        lambda x: ', '.join(x.dropna().astype(str)), axis=1)

all_info = pd.concat([c, e, a, f["All"]], axis=1)    
all_info.columns = ['Num_feature', 'Mean_Squared_Error', 'Accuracy', 'Feature']    
all_info = all_info.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)

all_info.to_csv("RRS_total_subset_RF_accuracy.csv", index=False)
f.to_csv("RRS_total_subset_RF.csv")
with open("RRS_total_RF_models.txt", "wb") as fp:
    pickle.dump(all_model, fp)

###############################################################################
############# Step 4.2 - Run forward algorithm + Ridge_regression #############
###############################################################################

from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.linear_model import ElasticNet, LarsCV, Lasso, LassoLars
from sklearn.linear_model import MultiTaskElasticNet, MultiTaskLasso
from sklearn.kernel_ridge import KernelRidge

y = rumi_meta["RRS_Brooding"]
rumi_region_plus = pd.concat([rumi_meta[['CurrentDepression', 'TIV', 'Age','Gender_1_male']], 
                              rumi_region], axis=1)
X_train, X_test, y_train, y_test = train_test_split(rumi_region_plus, y, test_size=0.3, random_state=42)

"""
alphas = np.logspace(-5, 5, 100)
tuned_parameters = [{"alpha": alphas}]
my_cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
model = Lasso()
"""

alphas = np.logspace(-5, 5, 100)
kernel = ["linear", "poly", "rbf", "sigmoid", "chi2", "laplacian"]
gamma = list(np.logspace(-2, 2, 100))
gamma.append("scale")
gamma.append("auto")
hyperparameter = {"alpha": alphas, 
               "kernel": kernel,
               "gamma": gamma}
my_cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
model = KernelRidge()

scoring = "neg_mean_squared_error"
n_selected_features = 240

F = []
count = 0
ddict = {}
all_F = []
all_c = []
all_acc = []
all_mse = []
all_model = []
start = time.time()
while count < n_selected_features:
    max_acc = 0
    min_err = np.inf
    time_loop = time.time()
    
    for i in X_train.columns:
            if i not in F:
                F.append(i)
                X_train_tmp = X_train[F]
                acc = 0
                #gsearch_cv = GridSearchCV(estimator = model, param_grid = tuned_parameters, 
                #                  scoring = "neg_mean_squared_error", cv = my_cv, n_jobs=-1)
                rsearch_cv = RandomizedSearchCV(estimator = model, param_distributions = hyperparameter, cv = my_cv,
                                        scoring = "neg_mean_squared_error", n_iter = 30, n_jobs = -1)
                #gsearch_cv.fit(X_train_tmp, y_train)
                #best_estimator = gsearch_cv.best_estimator_
                rsearch_cv.fit(X_train_tmp, y_train)
                best_estimator = rsearch_cv.best_estimator_
                y_pred = best_estimator.predict(X_test[F])
                mse = mean_squared_error(y_test, y_pred)
                #acc = metrics.accuracy_score(y_test, y_pred)
                F.pop()
                if mse < min_err:
                    min_err = mse
                    idx = i
                    best_model = best_estimator
                    #errors = abs(y_pred - y_test)
                    #mean_err = np.stack(errors/y_test)
                    #mean_err = mean_err[np.isfinite(mean_err)]
                    mape = mean_absolute_percentage_error(y_test, y_pred)
                    max_acc = 100 - (100*mape)
                    
    F.append(idx)
    count += 1
        
    print("The current number of features: {} - MSE: {}".format(count, round(min_err, 2)))
    print("Time for computation: {}".format(time.time() - time_loop))

    all_F.append(np.array(F))
    all_c.append(count)
    all_acc.append(max_acc)
    all_model.append(best_model)
    all_mse.append(min_err)

c = pd.DataFrame(all_c)
a = pd.DataFrame(all_acc)
f = pd.DataFrame(all_F)  
e = pd.DataFrame(all_mse)  
f["All"] = f[f.columns[0:]].apply(
        lambda x: ', '.join(x.dropna().astype(str)), axis=1)

all_info = pd.concat([c, e, a, f["All"]], axis=1)    
all_info.columns = ['Num_feature', 'Mean_Squared_Error', 'Accuracy', 'Feature']    
all_info = all_info.sort_values(by='Mean_Squared_Error', ascending=True).reset_index(drop=True)


# =============================================================================
# Test accuracy model 
# =============================================================================

from scipy.stats import spearmanr
from scipy import stats
import seaborn as sns
import pingouin as pt

all_features_grid = pd.read_csv("RF_subset_multi.csv")
all_info_grid = pd.read_csv("RF_subset_accuracy_multi.csv")
with open("RF_models_multi.txt", "rb") as fp:
    load_grid_model = pickle.load(fp)
subset = all_features_grid.drop(columns = ["Unnamed: 0"])

best_model_55 = load_grid_model[36]
subset = subset.iloc[36].dropna()
region_subset = ress_BPD_brain[subset]

X_train, X_test, y_train, y_test = train_test_split(region_subset, y, test_size=0.3, random_state=42)

evaluate_rf = automl.evaluate_multiclass(best_model_55, X_train, y_train, X_test, y_test,
                            model = "Random Forest", num_class=2, top_features=90, class_name = class_name)


best_model_55.fit(X_train, y_train)
y_pred = best_model_55.predict(X_test)

errors = abs(y_pred - y_test)
mean_err = np.stack(errors/y_test)
mean_err = mean_err[np.isfinite(mean_err)]
mape = 100 * np.mean(mean_err)
acc = 100 - mape

dt = {"True RRS_Brooding": y_test, "Predicted RRS_Brooding": y_pred}
df = pd.DataFrame(dt)
g = sns.lmplot(x="True RRS_Brooding", y="Predicted RRS_Brooding", data=df)
g.set(ylim = (min(y_test), max(y_test)))
g.set(xlim = (min(y_test), max(y_test)))
plt.text(-3.9, max(y_test) - 1, r'MSE = %.2f' % (mean_squared_error(y_test, y_pred)))
plt.text(-3.9, max(y_test) - 2, r'Corr = %.2f' % (spearmanr(y_test, y_pred)[0]))
#plt.text(-6, max(y_test) - 3, r'pval = %.2f' % (spearmanr(y_test, y_pred)[1]))


plt.scatter(y_pred, y_test, s=8)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--k')
plt.xlim(min(y_test), max(y_test))
plt.ylim(min(y_test), max(y_test))
plt.ylabel('True RRS_Brooding')
plt.xlabel('Predicted RRS_Brooding')
#plt.text(s='Random Forest without Forward varible', x=1,
#            y=2, fontsize=12, multialignment='center')
#plt.text(min(y_pred), max(y_test) - 1, r'$R^2$ = %.2f' % (r2_score(y_test, y_pred)))
plt.text(-4, max(y_test) - 1, r'MSE = %.2f' % (mean_squared_error(y_test, y_pred)))
plt.text(min(y_pred), max(y_test) - 11, r'Accuracy = %.2f' % acc)

importances = best_model_55.feature_importances_
indices = np.argsort(importances)[::-1]

feature_tab = pd.DataFrame({"Features": list(X_train.columns),
                            "Importance": importances})
feature_tab = feature_tab.sort_values("Importance", ascending = False).reset_index(drop=True)
index = feature_tab["Features"].iloc[:26]
importance_desc = feature_tab["Importance"].iloc[:26]
feature_space = []
for i in range(indices.shape[0]-1, -1, -1):
    feature_space.append(X_train.columns[indices[i]])
    
fig, ax = plt.subplots(figsize=(20,20))
ax = plt.gca()
plt.title("Feature importances", fontsize=30)
plt.barh(index, importance_desc, align="center", color="blue", alpha=0.6)
plt.grid(axis="x", color="white", linestyle="-")
plt.xlabel("The average of decrease in impurity", fontsize=20)
plt.ylabel("Features", fontsize=20)
plt.yticks(fontsize=30)
plt.xticks(fontsize=20)
plt.show()

RRS_region_plus = pd.concat([rumi_meta["RRS_Total"], region_subset], axis=1)

RRS_corr = RRS_region_plus.corr(method = "spearman").sort_values(by = "RRS_Total", ascending=False)
RRS_corr = RRS_corr["RRS_Total"]

sns.jointplot(data = RRS_region_plus, y = "RRS_Total", x = "BNA067lPCLA4ll", kind = "reg")

##

from scipy.stats import spearmanr

BPD_subset = pd.read_csv("BPD_19.csv")
MDD_feature = pd.read_csv("Feature_Importance_MDD.csv")
HC_feature = pd.read_csv("Feature_Importace_HC.csv")
All_feature = pd.read_csv("RSS_Brooding_subset_20.csv")

BPD_MDD_feature = MDD_feature[MDD_feature.index.isin(BPD_subset.columns)]

MDD_subset = ress_MDD_brain[MDD_feature.index]
HC_subset = ress_HC_brain[HC_feature.index]

BPD_subset_corr = pd.concat([ress_BPD_meta["RRS_Brooding"], BPD_subset], axis=1)
corr, pval = spearmanr(BPD_subset_corr)
pval = pd.DataFrame(pval).set_index(BPD_subset_corr_.index)
BPD_subset_corr_ = BPD_subset_corr.corr(method = "spearman").sort_values(by = "RRS_Brooding", ascending=False)
BPD_subset_corr_ = pd.concat([BPD_subset_corr_["RRS_Brooding"], pval[0]], axis=1) 
BPD_subset_corr_ = BPD_subset_corr_.drop("RRS_Brooding", axis=0)
BPD_subset_corr_.to_csv("Selected_BPD_corr.csv")

 
MDD_subset_corr = pd.concat([ress_MDD_meta["RRS_Brooding"], MDD_subset], axis=1)
MDD_subset_corr_ = MDD_subset_corr.corr(method = "spearman").sort_values(by = "RRS_Brooding", ascending=False)
corr, pval = spearmanr(MDD_subset_corr)
pval = pd.DataFrame(pval).set_index(MDD_subset_corr_.index)
MDD_subset_corr_ = pd.concat([MDD_subset_corr_["RRS_Brooding"], pval[0]], axis=1)
MDD_subset_corr_ = MDD_subset_corr_.drop("RRS_Brooding", axis=0)
MDD_subset_corr_.to_csv("Selected_MDD_corr.csv")


HC_subset_corr = pd.concat([ress_HC_meta["RRS_Brooding"], HC_subset], axis=1)
HC_subset_corr_ = HC_subset_corr.corr(method = "spearman").sort_values(by = "RRS_Brooding", ascending=False)
corr, pval = spearmanr(HC_subset_corr)
pval = pd.DataFrame(pval).set_index(HC_subset_corr_.index)
HC_subset_corr_ = pd.concat([HC_subset_corr_["RRS_Brooding"], pval[0]], axis=1)
HC_subset_corr_ = HC_subset_corr_.drop("RRS_Brooding", axis=0)
HC_subset_corr_.to_csv("Selected_HC_corr.csv")

All_subset_corr = pd.concat([ress_all_meta["RRS_Brooding"], all_subset], axis=1)
All_subset_corr_ = All_subset_corr.corr(method = "spearman").sort_values(by = "RRS_Brooding", ascending=False)
corr, pval = spearmanr(All_subset_corr)
pval = pd.DataFrame(pval).set_index(All_subset_corr_.index)
All_subset_corr_ = pd.concat([All_subset_corr_["RRS_Brooding"], pval[0]], axis=1)
All_subset_corr_ = All_subset_corr_.drop("RRS_Brooding", axis=0)
All_subset_corr_.to_csv("Selected_All_corr.csv")

All_subset_corr = All_feature.corr(method = "spearman").sort_values(by = "RRS_Brooding", ascending = False)
All_subset_corr = All_subset_corr.drop("RRS_Brooding", axis=0)
All_subset_corr = All_subset_corr["RRS_Brooding"]


MDD_tha = MDD_feature.loc[['BNA231lThamPFtha', 'BNA242rThaOtha', 'BNA244rThacTtha', 'BNA240rThaPPtha']]
BPD_tha = ress_BPD_brain[['BNA245lThalPFtha', 'BNA243lThacTtha', 'BNA234rThamPMtha', 'BNA236rThaStha']]
HC_tha = HC_feature.loc[["BNA242rThaOtha", "BNA232rThamPFtha", "BNA239lThaPPtha"]]

MDD_cin = MDD_feature.loc[['BNA186rCingA23c', 'BNA218rHippcHipp']]
HC_cin = HC_feature.loc[['BNA187lCingA32sg', 'BNA184rCingA24cd', 'BNA217lHippcHipp']]

MDD_fjg = MDD_feature.loc[['BNA030rIFGA44d']]


tha_3types = pd.concat([MDD_tha, HC_tha, MDD_cin, HC_cin], axis=0)

fig, ax = plt.subplots(figsize=(20,20))
ax = plt.gca()
plt.title("Feature importances", fontsize=30)
barlist = plt.barh(tha_3types.index, tha_3types["Importance"], align="center", color="blue", edgecolor='black', alpha=0.4)
for i in range(0,7):
    barlist[i].set_hatch('x')
for i in range(4):
    barlist[i].set_color('r')
for i in range(7,12):
    barlist[i].set_hatch('+')
barlist[7].set_color('r')
barlist[8].set_color('r')
plt.grid(axis="x", color="white", linestyle="-")
plt.xlabel("The important contribution", fontsize=20)
plt.ylabel("Features", fontsize=20)
plt.yticks(fontsize=30)
plt.xticks(fontsize=20)
plt.show()

important_brain_region = pd.concat([rumi_region_T, BPD_subset_corr_, 
                                    MDD_subset_corr_, HC_subset_corr_, All_subset_corr ], axis=1)
important_brain_region = important_brain_region.drop(columns = ['U041E1'])
important_brain_region.columns = ['BPD', 'MDD', 'HC', 'All']
important_brain_region.to_csv("Importance_brain_region.csv")

# =============================================================================
# Test code AutoML-FVS 
# =============================================================================



# =============================================================================
# Test code AutoML-FVS 
# =============================================================================

from FVS_algorithm import AutoML_FVS

fvs = AutoML_FVS()

all_info, all_model, f = fvs.RF_FVS(X_train, y_train, X_test, y_test)

all_info, all_model, f = fvs.KernelRidge_FVS(X_train, y_train, X_test, y_test)


















































