# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 10:07:09 2020

@author: biomet
"""

import numpy as np
import pandas as pd

from joblib import Parallel
from joblib import delayed

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from scipy.stats import norm 

import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
from sklearn.model_selection import train_test_split
import pickle 
import os, sys
import PCA_Analysis as pca
import RF_Analysis_Multiclass as rfc
import RF_Analysis_Binary as rfb
from Auto_ML_Multiclass import AutoML_classification

# =============================================================================
# Read data set
# ============================================================================= 

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
# Step 1 - Run Auto_ML
# ============================================================================= 

automl = AutoML_classification()

result = automl.fit(X_train, y_train, X_test, y_test)

# =============================================================================
# Step 2 - Run selected models
# ============================================================================= 

rf_best, _, _, _, _ = automl.Random_Forest(X_train, y_train, X_test, y_test)
evaluate_rf = automl.evaluate_multiclass(rf_best, X_train, y_train, X_test, y_test,
                            model = "Random Forest", num_class=3, top_features=20, class_name = class_name)

# =============================================================================
# Step 3 - Pre-screen features 
# ============================================================================= 

rf_best.fit(X_train, y_train)
variable_important = rf_best.feature_importances_
dict_important = {"Feature_name": X_train.columns,
                  "Importance": variable_important}
df_importance = pd.DataFrame(data = dict_important).sort_values(
                by="Importance", ascending=False).reset_index(drop=True)
#df_importance_119 = df_importance[df_importance["Feature_name"].isin(microbiome_subset.columns)]

"""
vimp = []
num_permutation = 100
for i in range(num_permutation):
    print(i)
    y_permuted = np.random.permutation(y_train)
    rf_best.fit(X_train, y_permuted)
    var_imp = rf_best.feature_importances_
    vimp.append(var_imp)
"""

def per_cal(rf, X_train, y_train, num_per):
    
    y_per = np.random.permutation(y_train)
    clf = rf
    clf.fit(X_train,y_per)
    var_imp = clf.feature_importances_
    
    return var_imp

num_permutation = 100
vimp = Parallel(n_jobs=4)(delayed(per_cal)(
    rf_best, X_train, y_train, num_per
    )for num_per in range(num_permutation))

df_vimp = pd.DataFrame(data = vimp).T

p_val = []
for i in range(len(df_vimp)):
    count = sum(df_vimp.iloc[i, :] >= df_importance.iloc[i, 1])
    p_var = (count + 1) / (num_permutation + 1)
    p_val.append(p_var)

dict_p_import = {"Feature_name": X_train.columns, 
                "Importance": variable_important,
                "P_value": p_val}

df_pval_important = pd.DataFrame(data=dict_p_import).sort_values(by="P_value", ascending=True).reset_index(drop=True)
df_pval_important.to_csv("OTU_pvalue.csv", index=False)
df_pval_important_new = df_pval_important[
    df_pval_important["P_value"] < 0.1].sort_values(by="Importance", ascending=False).reset_index(drop=True)
df_pval_important_new.to_csv("Selected_OTU_pvalue.csv", index=False)

selected_train_features = X_train[df_pval_important_new["Feature_name"]]
selected_test_features = X_test[df_pval_important_new["Feature_name"]]

# =============================================================================
# Step 4 - Run forward algorithm
# ============================================================================= 

import itertools
from scipy import interp
from itertools import cycle
from joblib import Parallel
from joblib import delayed
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

def run_forward(X_train, y_train, F, i, max_acc, base_model_rf, hyperparameter,
                n_iter_search, scoring):
    
    if i not in F:
        F.append(i)
        X_train_tmp = X_train[F]
        acc = 0
        rsearch_cv = RandomizedSearchCV(estimator=base_model_rf,
                                        random_state=42,
                                        param_distributions=hyperparameter,
                                        n_iter=n_iter_search,
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
            
    return idx, max_acc, best_model

X_train = selected_train_features
X_test = selected_test_features

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
n_selected_features = 30
    
# selected feature set, initialized to be empty
F = []
count = 0
ddict = {}
all_F = []
all_c = []
all_acc = []
all_model = []
while count < n_selected_features:
    max_acc = 0
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

    """
    idx, max_acc, best_model = Parallel(n_jobs=80)(delayed(run_forward)(
        X_train, y_train, F, i, max_acc, base_model_rf, hyperparameter,n_iter_search, scoring
        ) for i in X_train.columns)
    """
    F.append(idx)
    count += 1
        
    print("The current number of features: {} - Accuracy: {}%".format(count, round(max_acc*100, 2)))

    all_F.append(np.array(F))
    all_c.append(count)
    all_acc.append(max_acc)
    all_model.append(best_model)
    
c = pd.DataFrame(all_c)
a = pd.DataFrame(all_acc)
f = pd.DataFrame(all_F)    
f["All"] = f[f.columns[0:]].apply(
        lambda x: ', '.join(x.dropna().astype(str)), axis=1)
    
all_info = pd.concat([c, a, f["All"]], axis=1)    
all_info.columns = ['Num_feature', 'Accuracy', 'Feature']    
all_info = all_info.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)

all_info.to_csv("CDI_Permutation_124_subset_accuracy.csv", index=False)
f.to_csv("CDI_Permutation_124_subset.csv")
with open("CDI_Permutation_124_models.txt", "wb") as fp:
    pickle.dump(all_model, fp)


all_features_grid = pd.read_csv("CDI_Permutation_124_subset.csv")
all_info_grid = pd.read_csv("CDI_Permutation_124_subset_accuracy.csv")
with open("CDI_Permutation_124_models.txt", "rb") as fp:
    load_grid_model = pickle.load(fp)
subset = all_features_grid.drop(columns = ["Unnamed: 0", "All"])

best_model_16 = load_grid_model[16]
subset = subset.iloc[16].dropna()
microbiome_subset = microbiome[subset]

X_train, X_test, y_train, y_test = train_test_split(microbiome_subset, y, test_size=0.3, random_state=42)

evaluate_rf = automl.evaluate_multiclass(best_model_16, X_train, y_train, X_test, y_test,
                            model = "Random Forest", num_class=3, top_features=17, class_name = class_name)


X = pca.standardize(microbiome_subset)
pca_result = pca.pca_vis(X,y)
pca_full = pca.pca_redu(X, num_components = 30)








