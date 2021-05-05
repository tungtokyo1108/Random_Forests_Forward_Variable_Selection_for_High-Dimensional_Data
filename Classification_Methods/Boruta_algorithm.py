# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 12:06:45 2020

@author: biomet
"""

import numpy as np
import pandas as pd
import scipy as sp
from statsmodels.stats.multitest import fdrcorrection

import itertools
from scipy import interp
from itertools import cycle
from sklearn.utils import check_random_state, check_X_y
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV

import pickle 
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
# Main function 
# =============================================================================

def _get_importance_value(X_train, y_train, n_estimators):
    """
    

    Parameters
    ----------
    X_train : TYPE
        DESCRIPTION.
    y_train : TYPE
        DESCRIPTION.

    Returns
    -------
    imp : TYPE
        DESCRIPTION.

    """
    
    """
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
    n_folds = 10
    my_cv = TimeSeriesSplit(n_splits = n_folds).split(X_train)
    base_model_rf = RandomForestClassifier(random_state=42)
    rsearch_cv = RandomizedSearchCV(estimator=base_model_rf, 
                                   random_state=42,
                                   param_distributions=hyperparameter,
                                   n_iter=30,
                                   cv=my_cv,
                                   scoring="f1_macro",
                                   n_jobs=-1)
    rsearch_cv.fit(X_train, y_train)
    rb_best = rsearch_cv.best_estimator_
    rb_best.fit(X_train, y_train)
    imp = rb_best.feature_importances_
    """
    
    clf = RandomForestClassifier(n_estimators = n_estimators, random_state=42)
    clf.fit(X_train, y_train)
    imp = clf.feature_importances_
        
    return imp

def _get_tree_num(n_feat):
    depth = 10
    f_repr = 100
    multi = ((n_feat * 2) / (np.sqrt(n_feat * 2) * depth))
    n_estimators = int(multi * f_repr)
    return n_estimators

def _get_shuffle(seq):
    
    random_state = check_random_state(42)
    random_state.shuffle(seq)
    return seq

def _add_shadows_get_imps(X_train, y_train, dec_reg):
    
    """
    Expands the information system with newly built random attributes 
    and calculates the importance value
    
    Parameters
    ----------
    X_train : array-like
        The training input samples.
    y_train : array-like
        The target values.
    dec_reg : array-like
        Holds the decision about each feature
        0 - default state = tentative in orginal code
        1 - accepted in original code
       -1 - rejected in original code

    Returns
    -------
    imp_real : The importance value of real values
    imp_sha : The importance value of shadow values

    """
    
    # find features that tentative still 
    x_cur_ind = np.where(dec_reg >= 0)[0]
    x_cur = np.copy(X_train[:, x_cur_ind])
    x_cur_w = x_cur.shape[1]
    
    x_sha = np.copy(x_cur)
    # There must be at least 5 random attributes
    while (x_sha.shape[1] < 5):
        x_sha = np.hstack((x_sha, x_sha))
        
    # Now, we permute values in each attribute
    x_sha = np.apply_along_axis(_get_shuffle, 0, x_sha)
    
    not_rejected = np.where(dec_reg >= 0)[0].shape[0]
    n_tree = _get_tree_num(not_rejected)
    
    # Get importance values from new shadow input data
    imp = _get_importance_value(np.hstack((x_cur, x_sha)), y_train, 500)
    
    # Separate importances value of real and shadow features 
    imp_sha = imp[x_cur_w:]
    imp_real = np.zeros(X_train.shape[1])
    imp_real[:] = np.nan
    imp_real[x_cur_ind] = imp[:x_cur_w]
    
    return imp_real, imp_sha

def _assign_hits(hit_reg, cur_imp, imp_sha_max):
    
    """
    Register which the importance value of features is more than 
    the max value of shadows
    
    """
    cur_imp_no_nan = cur_imp[0]
    cur_imp_no_nan[np.isnan(cur_imp_no_nan)] = 0
    hits = np.where(cur_imp_no_nan > imp_sha_max)[0]
    hit_reg[hits] += 1
    
    return hit_reg

def _fdrcorrection(pvals, alpha=0.05):
    """
    Benjamini/Hochberg p-value correction for false discovery rate
    in statsmodels package
    """
    
    pvals = np.asarray(pvals)
    pvals_sortind = np.argsort(pvals)
    pvals_sorted = np.take(pvals, pvals_sortind)
    nobs = len(pvals_sorted)
    ecdffactor = np.arange(1, nobs+1) / float(nobs)
    
    reject = pvals_sorted <= ecdffactor * alpha
    if reject.any():
        rejectmax = max(np.nonzero(reject)[0])
        reject[:rejectmax] = True
    
    pvals_corrected_raw = pvals_sorted / ecdffactor
    pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
    pvals_corrected[pvals_corrected > 1] = 1
    
    # Reorder p-values and rejection mask to original order of pvals
    pvals_corrected_ = np.empty_like(pvals_corrected)
    pvals_corrected_[pvals_sortind] = pvals_corrected
    reject_ = np.empty_like(reject)
    reject_[pvals_sortind] = reject
    
    return reject_, pvals_corrected_

def _nan_rank_data(X, axis=1):
    """
    Replaces bottleneck's nanrankdata with scipy and numpy alternative

    """
    ranks = sp.stats.mstats.rankdata(X, axis=axis)
    ranks[np.isnan(X)] = np.nan
    
    return ranks

def _do_tests(dec_reg, hit_reg, runs, two_step = False, alpha = 0.05):
    
    active_features = np.where(dec_reg >= 0)[0]
    hits = hit_reg[active_features]
    two_step = two_step
    alpha = alpha
    
    to_accept_ps = sp.stats.binom.sf(hits - 1, runs, .5).flatten()
    to_reject_ps = sp.stats.binom.cdf(hits, runs, .5).flatten()
    
    if two_step:
        
        #to_accept = _fdrcorrection(to_accept_ps, alpha=0.05)[0]
        #to_reject = _fdrcorrection(to_reject_ps, alpha=0.05)[0]
        """
        pvalue correction for false discovery rate 
        Benjamini/Hochberg for independent or positive correlated
        """
        to_accept = fdrcorrection(to_accept_ps, alpha=0.05)[0]
        to_reject = fdrcorrection(to_reject_ps, alpha=0.05)[0]
        
        to_accept2 = to_accept_ps <= alpha / float(runs)
        to_reject2 = to_reject_ps <= alpha / float(runs)
        
        to_accept *= to_accept2
        to_reject *= to_reject2
    else:
        
        to_accept = to_accept_ps <= alpha / float(len(dec_reg))
        to_reject = to_reject_ps <= alpha / float(len(dec_reg))
        
    to_accept = np.where((dec_reg[active_features] == 0) * to_accept)[0]
    to_reject = np.where((dec_reg[active_features] == 0) * to_reject)[0]
    
    dec_reg[active_features[to_accept]] = 1
    dec_reg[active_features[to_reject]] = -1
    
    return dec_reg

def _print_results(dec_reg, runs, max_runs, flag):
    
    n_iter = str(runs) + '/' + str(max_runs)
    n_confirmed = np.where(dec_reg == 1)[0].shape[0]
    n_rejected = np.where(dec_reg == -1)[0].shape[0]
    cols = ['Iteration: ', 'Confirmed: ', 'Tentative: ', 'Rejected: ']
    
    if flag == 0:
        n_tentative = np.where(dec_reg == 0)[0].shape[0]
        content = map(str, [n_iter, n_confirmed, n_tentative, n_rejected])
        output = '\n'.join([x[0] + '\t' + x[1] for x in zip(cols, content)])
    
    print(output)

# =============================================================================
# Main part of Boruta algorithm
# =============================================================================

X_train, y_train = check_X_y(X_train, y_train)    
n_sample, n_feature = X_train.shape
runs = 1
max_runs = 100
perc = 50
dec_reg = np.zeros(n_feature, dtype=np.int)
hit_reg = np.zeros(n_feature, dtype=np.int)
imp_history = np.zeros(n_feature, dtype=np.float)
sha_max_history = []

while np.any(dec_reg == 0) and runs < max_runs:
    
    cur_imp = _add_shadows_get_imps(X_train, y_train, dec_reg)
    imp_sha_max = np.percentile(cur_imp[1], perc)    
  
    sha_max_history.append(imp_sha_max)
    imp_history = np.vstack((imp_history, cur_imp[0]))

    hit_reg = _assign_hits(hit_reg, cur_imp, imp_sha_max)  

    dec_reg = _do_tests(dec_reg, hit_reg, runs, two_step=True)
    
    _print_results(dec_reg, runs, max_runs, 0)
    
    runs += 1

confirmed = np.where(dec_reg == 1)[0]
tentative = np.where(dec_reg == 0)[0]

tentative_median = np.median(imp_history[1:, tentative], axis=0)
tentative_confirmed = np.where(tentative_median > np.median(sha_max_history))[0]
tentative = tentative[tentative_confirmed]

n_features_ = confirmed.shape[0]
support_ = np.zeros(n_feature, dtype=np.bool)
support_[confirmed] = 1
support_weak_ = np.zeros(n_feature, dtype=np.bool)
support_weak_[tentative] = 1

ranking_ = np.ones(n_feature, dtype=np.int)
ranking_[tentative] = 2
selected = np.hstack((confirmed, tentative))
not_selected = np.setdiff1d(np.arange(n_feature), selected)
imp_history_rejected = imp_history[1:, not_selected] * -1

iter_rank = _nan_rank_data(imp_history_rejected, axis=1)
rank_medians = np.nanmedian(iter_rank, axis=0)
ranks = _nan_rank_data(rank_medians, axis=0)
if tentative.shape[0] > 0:
    ranks = ranks - np.min(ranks) + 3
else:
    ranks = ranks - np.min(ranks) + 2
ranking_[not_selected] = ranks

indicies = support_
X_train_selected = X_train.iloc[:, indicies]
X_test_selected = X_test.iloc[:, indicies]


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

X_train = X_train_selected
X_test = X_test_selected

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
n_selected_features = 7400
    
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

all_info.to_csv("CDI_Boruta_1008_subset_accuracy.csv", index=False)
f.to_csv("CDI_Boruta_1008_subset.csv")
with open("CDI_Boruta_1008_models.txt", "wb") as fp:
    pickle.dump(all_model, fp)


all_features_grid = pd.read_csv("CDI_Boruta_1008_subset.csv")
all_info_grid = pd.read_csv("CDI_Boruta_1008_subset_accuracy.csv")
with open("CDI_Boruta_1008_models.txt", "rb") as fp:
    load_grid_model = pickle.load(fp)
subset = all_features_grid.drop(columns = ["Unnamed: 0", "All"])

best_model_96 = load_grid_model[95]
subset = subset.iloc[95].dropna()
microbiome_subset = microbiome[subset]

X_train, X_test, y_train, y_test = train_test_split(microbiome_subset, y, test_size=0.3, random_state=42)

evaluate_rf = automl.evaluate_multiclass(best_model_96, X_train, y_train, X_test, y_test,
                            model = "Random Forest", num_class=3, top_features=20, class_name = class_name)

X = pca.standardize(microbiome_subset)
pca_result = pca.pca_vis(X,y)
pca_full = pca.pca_redu(X, num_components = 30)













