#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 15:46:16 2019

@author: tungutokyo
"""

import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None
pd.set_option("display.max_columns", 60)

import matplotlib.pyplot as plt
import seaborn as sns

import itertools
from scipy import interp
from scipy.stats import fisher_exact
from itertools import cycle
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold

def get_RandSearchCV(X_train, y_train, X_test, y_test, scoring, type_search, output_file):
    from sklearn.model_selection import TimeSeriesSplit
    from datetime import datetime as dt 
    st_t = dt.now()
    # Numer of trees are used
    n_estimators = [5, 10, 50, 100, 150, 200, 250, 300]
    #n_estimators = list(np.arange(100,1000,50))
    #n_estimators = [1000]
    
    # Maximum depth of each tree
    max_depth = [5, 10, 25, 50, 75, 100]
    
    # Minimum number of samples per leaf 
    min_samples_leaf = [1, 2, 4, 8, 10]
    
    # Minimum number of samples to split a node
    min_samples_split = [2, 4, 6, 8, 10]
    
    # Maximum numeber of features to consider for making splits
    max_features = ["auto", "sqrt", "log2", None]
    
    hyperparameter = {'n_estimators': n_estimators,
                      'max_depth': max_depth,
                      'min_samples_leaf': min_samples_leaf,
                      'min_samples_split': min_samples_split,
                      'max_features': max_features}
    
    cv_timeSeries = TimeSeriesSplit(n_splits=5).split(X_train)
    base_model_rf = RandomForestClassifier(criterion="gini", random_state=42)
    base_model_gb = GradientBoostingClassifier(criterion="friedman_mse", random_state=42)
    
    # Run randomzed search 
    n_iter_search = 30
    if type_search == "RandomSearchCV-RandomForest":
        rsearch_cv = RandomizedSearchCV(estimator=base_model_rf, 
                                   random_state=42,
                                   param_distributions=hyperparameter,
                                   n_iter=n_iter_search,
                                   cv=cv_timeSeries,
                                   scoring=scoring,
                                   n_jobs=-1)
    elif type_search == "RandomSearchCV-GradientBoosting":
        rsearch_cv = RandomizedSearchCV(estimator=base_model_gb, 
                                   random_state=42,
                                   param_distributions=hyperparameter,
                                   n_iter=n_iter_search,
                                   cv=cv_timeSeries,
                                   scoring=scoring,
                                   n_jobs=-1)
    
    rsearch_cv.fit(X_train, y_train)
    #f = open("output.txt", "a")
    print("Best estimator obtained from CV data: \n", rsearch_cv.best_estimator_, file=output_file)
    print("Best Score: ", rsearch_cv.best_score_, file=output_file)
    return rsearch_cv

def get_GridSearchCV(X_train, y_train, X_test, y_test, scoring, type_search, output_file):
    from sklearn.model_selection import TimeSeriesSplit
    from datetime import datetime as dt 
    st_t = dt.now()
    # Numer of trees are used
    n_estimators = [5, 10, 50, 100, 150, 200, 250, 300]
    #n_estimators = list(np.arange(100,1000,50))
    #n_estimators = [1000]
    
    # Maximum depth of each tree
    max_depth = [5, 10, 25, 50, 75, 100]
    
    # Minimum number of samples per leaf 
    min_samples_leaf = [1, 2, 4, 8, 10]
    
    # Minimum number of samples to split a node
    min_samples_split = [2, 4, 6, 8, 10]
    
    # Maximum numeber of features to consider for making splits
    max_features = ["auto", "sqrt", "log2", None]
    
    hyperparameter = {'n_estimators': n_estimators,
                      'max_depth': max_depth,
                      'min_samples_leaf': min_samples_leaf,
                      'min_samples_split': min_samples_split,
                      'max_features': max_features}
    
    cv_timeSeries = TimeSeriesSplit(n_splits=5).split(X_train)
    base_model_rf = RandomForestClassifier(criterion="gini", random_state=42)
    base_model_gb = GradientBoostingClassifier(criterion="friedman_mse", random_state=42)
    
    # Run Grid Search
    if type_search == "GridSearchCV-RandomForest":
        gsearch_cv = GridSearchCV(estimator = base_model_rf, 
                                  param_grid = hyperparameter, 
                                  cv = cv_timeSeries,
                                  scoring = scoring,
                                  # random_state = 42,
                                  n_jobs = -1)
    elif type_search == "GridSearchCV-GradientBoosting":
        gsearch_cv = GridSearchCV(estimator = base_model_gb, 
                                  param_grid = hyperparameter,
                                  cv = cv_timeSeries,
                                  scoring = scoring,
                                  # random_state = 42, 
                                  n_jobs = -1)
    
    gsearch_cv.fit(X_train, y_train)
    print("Best estimator obtained from CV data: \n", gsearch_cv.best_estimator_, file=output_file)
    print("Best Score: ", gsearch_cv.best_score_, file=output_file)
    
def performance(best_clf, X_train, y_train, X_test, y_test, type_search, output_file):
    #f = open("output.txt", "a")
    print("-"*100)
    print("~~~~~~~~~~~~~~~~~~ PERFORMANCE EVALUATION ~~~~~~~~~~~~~~~~~~~~~~~~", file=output_file)
    print("Detailed report for the {} algorithm".format(type_search), file=output_file)
    
    y_pred = best_clf.predict(X_test)
    y_pred_prob = best_clf.predict_proba(X_test)
    
    test_accuracy = accuracy_score(y_test, y_pred, normalize=True) * 100
    points = accuracy_score(y_test, y_pred, normalize=False)
    print("The number of accurate predictions out of {} data points on unseen data is {}".format(
            X_test.shape[0], points), file=output_file)
    print("Accuracy of the {} model on unseen data is {}".format(
            type_search, np.round(test_accuracy, 2)), file=output_file)
    
    print("Precision of the {} model on unseen data is {}".format(
            type_search, np.round(metrics.precision_score(y_test, y_pred), 4)), file=output_file)
    print("Recall of the {} model on unseen data is {}".format(
           type_search, np.round(metrics.recall_score(y_test, y_pred), 4)), file=output_file)
    print("F1 score of the {} model on unseen data is {}".format(
            type_search, np.round(metrics.f1_score(y_test, y_pred), 4)), file=output_file)
    
    print("\nClassification report for {} model: \n".format(type_search), file=output_file)
    print(metrics.classification_report(y_test, y_pred), file=output_file)
    
    plt.figure(figsize=(12,12))
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    cnf_matrix_norm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    print("\nThe Confusion Matrix: \n", file=output_file)
    print(cnf_matrix, file=output_file)
    
    cmap = plt.cm.Blues
    sns.heatmap(cnf_matrix_norm, annot=True, cmap=cmap, fmt=".2f", annot_kws={"size":15}, linewidths=.05)
    if type_search == "RandomSearchCV-RandomForest":
        plt.title("The Normalized Confusion Matrix - {}".format("RS-RandomForest"), fontsize=20)
    elif type_search == "RandomSearchCV-GradientBoosting":
        plt.title("The Normalized Confusion Matrix - {}".format("RS-GradientBoosting"), fontsize=20)
    elif type_search == "GridSearchCV-RandomForest":
        plt.title("The Normalized Confusion Matrix - {}".format("GS-RandomForest"), fontsize=20)
    elif type_search == "GridSearchCV-GradientBoosting":
        plt.title("The Normalized Confusion Matrix - {}".format("GS-GradientBoosting"), fontsize=20)
    
    plt.ylabel("True label", fontsize=15)
    plt.xlabel("Predicted label", fontsize=15)
    plt.show()
    
    print("\nROC curve and AUC")
    y_pred = best_clf.predict(X_test)
    y_pred_prob = best_clf.predict_proba(X_test)[:,1]
    
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
    plt.figure(figsize=(12,12))
    plt.plot(fpr, tpr, label="ROC curve with AUC = {} - Accuracy = {}%".format(round(
            metrics.roc_auc_score(y_test, y_pred_prob), 3), round(test_accuracy, 2)))
    plt.plot([0,1], [0,1], "k--", lw=2, color="r", label="Chance")
    plt.title("ROC-AUC for {}".format(type_search), fontsize=20)
    plt.xlabel("False Positive Rate", fontsize=15)
    plt.ylabel("True Positive Rate", fontsize=15)
    plt.legend(loc="lower right")
    plt.show()
    
    importances = best_clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    return {"importance": importances, 
            "index": indices,
            "y_pred": y_pred,
            "y_pred_prob": y_pred_prob}

def RF_classifier(X_train, y_train, X_test, y_test, scoring, type_search, output_file, top_feature):
    print("*"*100)
    print("Starting {} steps with {} for evaluation rules...".format(type_search, scoring))
    print("*"*100)
    
    if type_search == "RandomSearchCV-RandomForest":
        search_cv = get_RandSearchCV(X_train, y_train, X_test, y_test, scoring, type_search, output_file)
    elif type_search == "RandomSearchCV-GradientBoosting":
        search_cv = get_RandSearchCV(X_train, y_train, X_test, y_test, scoring, type_search, output_file)
    elif type_search == "GridSearchCV-RandomForest":
        search_cv = get_GridSearchCV(X_train, y_train, X_test, y_test, scoring, type_search, output_file)
    elif type_search == "GridSearchCV-GradientBoosting":
        search_cv = get_GridSearchCV(X_train, y_train, X_test, y_test, scoring, type_search, output_file)
        
    best_estimator = search_cv.best_estimator_
    max_depth = search_cv.best_estimator_.max_depth
    n_estimators = search_cv.best_estimator_.n_estimators
    var_imp_rf = performance(best_estimator, X_train, y_train, X_test, y_test, type_search, output_file)
    
    print("\n~~~~~~~~~~~~~ Features ranking and ploting ~~~~~~~~~~~~~~~~~~~~~\n", file=output_file)
    
    importances_rf = var_imp_rf["importance"]
    indices_rf = var_imp_rf["index"]
    y_pred = var_imp_rf["y_pred"]
        
    feature_tab = pd.DataFrame({"Features" : list(X_train.columns),
                                "Importance": importances_rf})
    feature_tab = feature_tab.sort_values("Importance", ascending = False).reset_index(drop=True)
    print(feature_tab, file=output_file)
    
    #index = np.arange(len(X_train.columns))
    #importance_desc = sorted(importances_rf)
    index = feature_tab["Features"].iloc[:top_feature]
    importance_desc = feature_tab["Importance"].iloc[:top_feature]
    feature_space = []
    for i in range(indices_rf.shape[0]-1, -1, -1):
        feature_space.append(X_train.columns[indices_rf[i]])
    
    fig, ax = plt.subplots(figsize=(15,15))
    ax = plt.gca()
    plt.title("Feature importances for {} Model".format(type_search), fontsize=20)
    plt.barh(index, importance_desc, align="center", color="blue", alpha=0.6)
    plt.grid(axis="x", color="white", linestyle="-")
    #plt.yticks(index, feature_space)
    plt.xlabel("The Average of Decrease in Impurity", fontsize=15)
    plt.ylabel("Features", fontsize=15)
    ax.tick_params(axis="both", which="both", length=0)
    plt.show()
    
    return {"Main Results": var_imp_rf,
            "Importance Features": feature_tab}

def prep_rf(df, H_smpls, dis_smpls, random_state):
    
    if not isinstance(H_smpls, list):
        H_smpls = list(H_smpls)
    if not isinstance(dis_smpls, list):
        dis_smpls = list(dis_smpls)
        
    all_smpls = H_smpls + dis_smpls
    
    rf = RandomForestClassifier(n_estimators = 1000, random_state = random_state)
    X = df.loc[all_smpls].values
    y = [1 if i in dis_smpls else 0 for i in all_smpls]
    return rf, X, y

def RF_SKF(rf, X, y, num_cv = 5, random_state=None, plot = True, disease="CDI"):
    
    if isinstance(y, list):
        y = np.asarray(y)
    
    #rf = RandomForestClassifier(n_estimators = n_estimators, random_state = random_state)
    
    skf = StratifiedKFold(n_splits = num_cv, shuffle = True, random_state = random_state)
    mean_tpr = 0
    mean_fpr = np.linspace(0, 1, 100)
    conf_mat = np.asarray([[0,0], [0,0]])
    tprs = []
    aucs = []
    accurancies = []
    kappa_scores = []
    test_accuracies = 0
    test_aver_pres = 0
    test_f1s = 0
    cv_count = 0
    
    if plot == True:
        plt.figure(figsize=(12,12))
    
    for train, test in skf.split(X,y):
        probas_ = rf.fit(X[train], y[train]).predict_proba(X[test])
        y_pred = rf.predict(X[test])
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        conf_mat += confusion_matrix(y[test], y_pred, labels = [0, 1])
        test_accurancy = metrics.accuracy_score(y[test], y_pred, normalize=True) * 100
        test_accuracies += test_accurancy
        accurancies.append(test_accurancy)
        test_aver_pre = metrics.average_precision_score(y[test], probas_[:, 1])
        test_aver_pres += test_aver_pre
        test_f1 = metrics.f1_score(y[test], y_pred)
        test_f1s += test_f1
        cohen_kappa_score = metrics.cohen_kappa_score(y[test], y_pred)
        kappa_scores.append(cohen_kappa_score)
        if plot == True:
            plt.plot(fpr, tpr, lw = 1, alpha = 0.3, 
                 label = "ROC fold %d (AUC = %0.2f - Accuracy = %0.2f)" % (cv_count, roc_auc, test_accurancy))
        cv_count += 1
    if plot == True:
        plt.plot([0,1], [0,1], linestyle = "--", lw = 2, color = "r", 
             label = "Chance", alpha = .8)
    
    mean_tpr = np.mean(tprs, axis = 0)
    mean_tpr[-1] = 1.0
    roc_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    test_accuracies /= cv_count 
    test_aver_pres /= cv_count
    test_f1s /= cv_count
    if plot == True:
        plt.plot(mean_fpr, mean_tpr, color = "b", 
             label = r'Mean ROC (AUC = %0.2f $\pm$ %0.2f) - Accuracy = %0.2f' % (roc_auc, std_auc, test_accuracies), 
             lw = 2, alpha = .8)
        
    std_tpr = np.std(tprs, axis = 0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    if plot == True:
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color = "grey", 
                     alpha = .2, label = r'$\pm$ 1 std. dev.')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel("False Positive Rate", fontsize=15)
        plt.ylabel("True Positive Rate", fontsize=15)
        plt.title("ROC for '{}' disease".format(disease), fontsize=20)
        plt.legend(loc = "lower right")
        plt.show()
    
    _, fisher_p = fisher_exact(conf_mat)
    
    return {i: j for i, j in 
            zip(("roc_auc", "conf_mat", "mean_fpr", "mean_tpr", "fisher_p", "kappa", 
                 "accurancy", "auc_pr", "f1"),
                (roc_auc, conf_mat, mean_fpr, mean_tpr, fisher_p, kappa_scores, 
                 test_accuracies, test_aver_pres, test_f1s))}

def RF_SKF_search(X, y, n_est, crit, max_depth, min_split, min_leaf, max_features, random_state, report_loop=True):
    
    print(n_est, crit, min_split, min_leaf)
    rf = RandomForestClassifier(n_estimators = n_est,
                                max_depth = max_depth,
                                criterion = crit,
                                min_samples_split = min_split,
                                min_samples_leaf = min_leaf,
                                max_features = max_features,
                                random_state = random_state)
    
    # Cross-validated results
    try:
        results = RF_SKF(rf, X, y, num_cv = 5, random_state=random_state, plot = False)
    except:
        results = {"roc_auc": np.nan, "fisher_p": np.nan}
        
    # Get oob_score for non-cross validated results
    rf = RandomForestClassifier(n_estimators = n_est, 
                                criterion = crit,
                                min_samples_split = min_split,
                                min_samples_leaf = min_leaf,
                                random_state = random_state,
                                oob_score = True)
    try:
        score = rf.fit(X, y).oob_score_
    except:
        score = np.nan
    
    if report_loop == True:    
        print("Finished. (AUC = {:.2f} and Accuracy = {:.2f})".format(results["roc_auc"], results["accurancy"]))
    
    return [n_est, crit, max_depth, min_split, min_leaf, max_features, 
            results["roc_auc"], results["fisher_p"], 
            results["accurancy"], results["auc_pr"], results["f1"], score]

def RF_SKF_run(X, y):
    
    # Numer of trees are used
    n_estimators = [5, 10, 50, 100, 150, 200, 250, 300]
    
    criterion = ["gini", "entropy"]  

    # Maximum depth of each tree
    max_depths = [5, 10, 25, 50, 75, 100]
    
    # Minimum number of samples per leaf 
    min_samples_leaf = [1, 2, 4, 8, 10]
    
    # Minimum number of samples to split a node
    min_samples_split = [2, 4, 6, 8, 10]
    
    # Maximum numeber of features to consider for making splits
    max_featuress = ["auto", "sqrt", "log2", None]
    
    random_state = 42
    
    rf_result_all = []
    for crit in criterion:
        for min_split in min_samples_split:
            for min_leaf in min_samples_leaf:
                for n_est in n_estimators:
                    for max_depth in max_depths:
                        for max_features in max_featuress:
                            rf_result = RF_SKF_search(X, y, n_est, crit, max_depth, 
                                           min_split, min_leaf, max_features, random_state)
                            rf_result_all.append(rf_result)
    
    rf_result_df = pd.DataFrame(rf_result_all, 
                            columns = ["n_estimators", "criterion", "max_depth",
                                       "min_samples_split", "min_samples_leaf", "max_features", 
                                       "roc_auc", "fisher_p", "Accurancy", "auc_prec_recall", "F1_score",
                                       "oob_score"]).sort_values("Accurancy", ascending = False).reset_index(drop=True)
        
    return rf_result_df










































