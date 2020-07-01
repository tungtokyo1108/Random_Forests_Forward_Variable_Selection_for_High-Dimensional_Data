#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 00:37:32 2019

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
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier

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
    else:
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

def performance_rand(best_clf, X_train, y_train, X_test, y_test, type_search, num_class, output_file, class_name):
    #f = open("output.txt", "a")
    print("-"*100)
    print("~~~~~~~~~~~~~~~~~~ PERFORMANCE EVALUATION ~~~~~~~~~~~~~~~~~~~~~~~~", file=output_file)
    print("Detailed report for the {} algorithm".format(type_search), file=output_file)
    
    best_clf.fit(X_train, y_train)
    y_pred = best_clf.predict(X_test)
    y_pred_prob = best_clf.predict_proba(X_test)
    
    test_accuracy = accuracy_score(y_test, y_pred, normalize=True) * 100
    points = accuracy_score(y_test, y_pred, normalize=False)
    print("The number of accurate predictions out of {} data points on unseen data is {}".format(
            X_test.shape[0], points), file=output_file)
    print("Accuracy of the {} model on unseen data is {}".format(
            type_search, np.round(test_accuracy, 2)), file=output_file)
    
    print("Precision of the {} model on unseen data is {}".format(
            type_search, np.round(metrics.precision_score(y_test, y_pred, average="macro"), 4)), file=output_file)
    print("Recall of the {} model on unseen data is {}".format(
           type_search, np.round(metrics.recall_score(y_test, y_pred, average="macro"), 4)), file=output_file)
    print("F1 score of the {} model on unseen data is {}".format(
            type_search, np.round(metrics.f1_score(y_test, y_pred, average="macro"), 4)), file=output_file)
    
    print("\nClassification report for {} model: \n".format(type_search), file=output_file)
    print(metrics.classification_report(y_test, y_pred), file=output_file)
    
    plt.figure(figsize=(12,12))
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    cnf_matrix_norm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    print("\nThe Confusion Matrix: \n", file=output_file)
    print(cnf_matrix, file=output_file)
    
    #class_name = ["CDI", "ignore-nonCDI", "Health"]
    #class_name = ["CRC", "Adenomas", "Health"]
    # class_name = ["OB", "OW", "Health"]
    class_name = class_name
    cmap = plt.cm.Blues
    plt.imshow(cnf_matrix_norm, interpolation="nearest", cmap=cmap)
    plt.colorbar()
    fmt = ".2g"
    thresh = cnf_matrix_norm.max()/2
    for i, j in itertools.product(range(cnf_matrix_norm.shape[0]), range(cnf_matrix_norm.shape[1])):
        plt.text(j,i,format(cnf_matrix_norm[i,j], fmt), ha="center", va="center", 
                 color="white" if cnf_matrix_norm[i,j] > thresh else "black", fontsize=35)
    
    plt.xticks(np.arange(num_class), labels = class_name, fontsize=30)
    plt.yticks(np.arange(num_class), labels = class_name, fontsize=30)
    
    
    plt.ylabel("True label", fontsize=30)
    plt.xlabel("Predicted label", fontsize=30)
    plt.ylim((num_class - 0.5, -0.5))
    plt.show()
    

    #plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)
    
    """
    cmap = plt.cm.Blues
    sns.heatmap(cnf_matrix_norm, annot=True, cmap=cmap, fmt=".2f", annot_kws={"size":15}, linewidths=.05)
    if type_search == "RandomSearchCV-RandomForest":
        plt.title("The Normalized Confusion Matrix - {}".format("RandomForest"), fontsize=20)
    else:
        plt.title("The Normalized Confusion Matrix - {}".format("GradientBoosting"), fontsize=20)
    
    plt.ylabel("True label", fontsize=15)
    plt.xlabel("Predicted label", fontsize=15)
    plt.show()
    """
    
    print("\nROC curve and AUC")
    y_pred = best_clf.predict(X_test)
    y_pred_prob = best_clf.predict_proba(X_test)
    y_test_cat = np.array(pd.get_dummies(y_test))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_class):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_test_cat[:,i], y_pred_prob[:,i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_class)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_class):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        
    mean_tpr /= num_class
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(12,12))
    plt.plot(fpr["macro"], tpr["macro"], 
         label = "macro-average ROC curve with AUC = {} - Accuracy = {}%".format(
                 round(roc_auc["macro"], 2), round(test_accuracy, 2)),
         color = "navy", linestyle=":", linewidth=4)
    colors = cycle(["red", "orange", "blue", "pink", "green"])
    for i, color in zip(range(num_class), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label = "ROC curve of class {0} (AUC = {1:0.2f})".format(i, roc_auc[i]))   
    plt.plot([0,1], [0,1], "k--", lw=2)
    plt.title("ROC-AUC for Random Forest".format(type_search), fontsize=20)
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

def RF_classifier(X_train, y_train, X_test, y_test, scoring, type_search, num_class, output_file, top_feature, class_name):
    #f = open("output.txt", "a")
    print("*"*100)
    print("Starting {} steps with {} for evaluation rules...".format(type_search, scoring))
    print("*"*100)
    
    rsearch_cv = get_RandSearchCV(X_train, y_train, X_test, y_test, scoring, type_search, output_file)
    
    best_estimator = rsearch_cv.best_estimator_
    max_depth = rsearch_cv.best_estimator_.max_depth
    n_estimators = rsearch_cv.best_estimator_.n_estimators
    var_imp_rf = performance_rand(best_estimator, X_train, y_train, X_test, y_test, type_search, 
                                  num_class, output_file, class_name)
    
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
    
    fig, ax = plt.subplots(figsize=(20,25))
    ax = plt.gca()
    plt.title("Feature importances of Random Forest".format(type_search), fontsize=30)
    plt.barh(index, importance_desc, align="center", color="blue", alpha=0.6)
    plt.grid(axis="x", color="white", linestyle="-")
    plt.yticks(fontsize=30)
    plt.xticks(fontsize=20)
    plt.xlabel("The Average of Decrease in Impurity", fontsize=20)
    plt.ylabel("Features", fontsize=30)
    ax.tick_params(axis="both", which="both", length=0)
    plt.show()
    
    return {"Main Results": var_imp_rf,
            "Importance Features": feature_tab}
    
###############################################################################
################### Stratified K-Folds cross-validator ########################
###############################################################################    
    
def RF_SKF(rf, X, y, num_cv = 5, random_state = 42):
    
    skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
    
    test_accuracies = 0
    test_precisions = 0
    test_recalls = 0
    test_f1s = 0
    cv_count = 0
    
    # rf = RandomForestClassifier(n_estimators = 100)
    
    for train, test in skf.split(X,y):
        probas_ = rf.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])
        y_pred = rf.predict(X.iloc[test])
        test_accuracy = metrics.accuracy_score(y.iloc[test], y_pred, normalize = True) * 100
        test_accuracies += test_accuracy
        test_precision = metrics.precision_score(y.iloc[test], y_pred, average="macro")
        test_precisions += test_precision
        test_recall_score = metrics.recall_score(y.iloc[test], y_pred, average="macro")
        test_recalls += test_recall_score
        test_f1_score = metrics.f1_score(y.iloc[test], y_pred, average="macro")
        test_f1s += test_f1_score
        cv_count += 1
        
    test_accuracies /= cv_count
    test_precisions /= cv_count
    test_recalls /= cv_count
    test_f1s /= cv_count
    
    return {i: j for i, j in 
            zip(("Accuracy", "Precision_Score", "Recall_Score", "F1_Score"),
                (test_accuracies, test_precisions, test_recalls, test_f1s))}

def RF_SKF_search(X, y, n_est, crit, max_depth, min_split, min_leaf, max_feature, 
                  num_cv = 5, random_state = 42, report_loop = True):
    
    print(n_est, crit, min_split, min_leaf)
    rf = RandomForestClassifier(n_estimators = n_est,
                                max_depth = max_depth,
                                criterion = crit,
                                min_samples_split = min_split,
                                min_samples_leaf = min_leaf,
                                max_features = max_feature,
                                random_state = random_state)
    
    # Cross_validated results
    try:
        results = RF_SKF(rf, X, y, num_cv = num_cv, random_state = random_state)
    except:
        results = {"Accuracy": np.nan}
        
    # Get oob_score for non-cross validated results
    rf = RandomForestClassifier(n_estimators = n_est, 
                                max_depth = max_depth,
                                criterion = crit,
                                min_samples_split = min_split,
                                min_samples_leaf = min_leaf,
                                max_features = max_feature,
                                random_state = random_state,
                                oob_score = True)
    
    try:
        score = rf.fit(X, y).oob_score_
    except:
        score = np.nan
    
    if report_loop == True:
        print("Finished. (Accuracy = {:.2f}%)".format(results["Accuracy"]))
        
    return [n_est, crit, max_depth, min_split, min_leaf, max_feature,
            results["Accuracy"], results["Precision_Score"], results["Recall_Score"], results["F1_Score"], score]
    
def RF_SKF_run(X, y, report_loop = True):
    
    # Numer of trees are used
    n_estimators = [50, 100, 150, 200, 250, 300]
    
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
                                           min_split, min_leaf, max_features, random_state, 
                                           report_loop = report_loop)
                            rf_result_all.append(rf_result)
    
    rf_result_df = pd.DataFrame(rf_result_all, 
                            columns = ["n_estimators", "criterion", "max_depth",
                                       "min_samples_split", "min_samples_leaf", "max_features", 
                                       "Accurancy", "Precision_Score", "Recall_Score", "F1_score",
                                       "oob_score"]).sort_values("Accurancy", ascending = False).reset_index(drop=True)
        
    return rf_result_df

def performance_SKF(best_clf, X_train, y_train, X_test, y_test, type_search, num_class, output_file):
    #f = open("output.txt", "a")
    print("-"*100)
    print("~~~~~~~~~~~~~~~~~~ PERFORMANCE EVALUATION ~~~~~~~~~~~~~~~~~~~~~~~~", file=output_file)
    print("Detailed report for the {} algorithm".format(type_search), file=output_file)
    
    best_clf.fit(X_train, y_train)
    y_pred = best_clf.predict(X_test)
    y_pred_prob = best_clf.predict_proba(X_test)
    
    test_accuracy = accuracy_score(y_test, y_pred, normalize=True) * 100
    points = accuracy_score(y_test, y_pred, normalize=False)
    print("The number of accurate predictions out of {} data points on unseen data is {}".format(
            X_test.shape[0], points), file=output_file)
    print("Accuracy of the {} model on unseen data is {}".format(
            type_search, np.round(test_accuracy, 2)), file=output_file)
    
    print("Precision of the {} model on unseen data is {}".format(
            type_search, np.round(metrics.precision_score(y_test, y_pred, average="macro"), 4)), file=output_file)
    print("Recall of the {} model on unseen data is {}".format(
           type_search, np.round(metrics.recall_score(y_test, y_pred, average="macro"), 4)), file=output_file)
    print("F1 score of the {} model on unseen data is {}".format(
            type_search, np.round(metrics.f1_score(y_test, y_pred, average="macro"), 4)), file=output_file)
    
    print("\nClassification report for {} model: \n".format(type_search), file=output_file)
    print(metrics.classification_report(y_test, y_pred), file=output_file)
    
    plt.figure(figsize=(12,12))
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    cnf_matrix_norm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    print("\nThe Confusion Matrix: \n", file=output_file)
    print(cnf_matrix, file=output_file)
    """
    cmap = plt.cm.Blues
    plt.imshow(cnf_matrix, interpolation="nearest", cmap=cmap)
    plt.colorbar()
    fmt = "d"
    thresh = cnf_matrix.max()/2
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j,i,format(cnf_matrix[i,j], fmt), ha="center", va="center", 
                 color="white" if cnf_matrix[i,j] > thresh else "black")
    """
    cmap = plt.cm.Blues
    sns.heatmap(cnf_matrix_norm, annot=True, cmap=cmap, fmt=".2f", annot_kws={"size":15}, linewidths=.05)
    plt.title("The Normalized Confusion Matrix - {}".format(type_search), fontsize=20)
    plt.ylabel("True label", fontsize=15)
    plt.xlabel("Predicted label", fontsize=15)
    plt.show()
    
    print("\nROC curve and AUC")
    y_pred = best_clf.predict(X_test)
    y_pred_prob = best_clf.predict_proba(X_test)
    y_test_cat = np.array(pd.get_dummies(y_test))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_class):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_test_cat[:,i], y_pred_prob[:,i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_class)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_class):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        
    mean_tpr /= num_class
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(12,12))
    plt.plot(fpr["macro"], tpr["macro"], 
         label = "macro-average ROC curve with AUC = {} - Accuracy = {}%".format(
                 round(roc_auc["macro"], 2), round(test_accuracy, 2)),
         color = "navy", linestyle=":", linewidth=4)
    colors = cycle(["red", "orange", "blue", "pink", "green"])
    for i, color in zip(range(num_class), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label = "ROC curve of class {0} (AUC = {1:0.2f})".format(i, roc_auc[i]))   
    plt.plot([0,1], [0,1], "k--", lw=2)
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

###############################################################################
################ Forward algorithm to variable selection ######################
###############################################################################


def random_forest_forward(X_train, y_train, X_test, y_test, n_selected_features = 1000, scoring='accuracy'):
    from sklearn.model_selection import TimeSeriesSplit
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
                      'max_features': max_features}
    
    cv_timeSeries = TimeSeriesSplit(n_splits=5).split(X_train)
    base_model_rf = RandomForestClassifier(criterion='gini', random_state=42)
    n_iter_search = 30
    scoring = scoring
    
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
    
    print("The total time for searching subset: {}".format(dt.now()-st_t))
    
    return all_info, all_model, f


def random_forest_randomforward(X_train, y_train, X_test, y_test, 
                                n_selected_features = 1000, scoring='accuracy', n_iter=1000):
    
    from sklearn.model_selection import TimeSeriesSplit
    from datetime import datetime as dt
    import random
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
                      'max_features': max_features}
    
    cv_timeSeries = TimeSeriesSplit(n_splits=5).split(X_train)
    base_model_rf = RandomForestClassifier(criterion='gini', random_state=42)
    n_iter_search = 30
    scoring = scoring
    
    # selected feature set, initialized to be empty
    count = 0
    ddict = {}
    all_F = []
    all_c = []
    all_acc = []
    all_model = []
    
    while count < n_selected_features:
        #F = []
        max_acc = 0
        for i in range(n_iter):
            col_train = random.sample(list(X_train.columns), count+1)
            col_train = np.array(col_train)
            X_train_tmp = X_train[col_train]
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
            y_pred = best_estimator.predict(X_test[col_train])
            acc = metrics.accuracy_score(y_test, y_pred)
            if acc > max_acc:
                max_acc = acc 
                idx = col_train
                best_model = best_estimator 
        #F.append(idx)
        count += 1
        
        print("The current number of features: {} - Accuracy: {}%".format(count, round(max_acc*100, 2)))
        
        all_F.append(idx)
        all_c.append(count)
        all_acc.append(max_acc)
        all_model.append(best_model)
    
    c = pd.DataFrame(all_c)
    a = pd.DataFrame(all_acc)
    f = pd.DataFrame(all_F)
    f["All"] = f[f.columns[0:]].apply(
            lambda x: ', '.join(x.dropna().astype(str)), axis=1)
    
    all_info = pd.concat([c, a, f["All"]], axis=1)
    all_info.columns = ['Num_features', 'Accuracy', 'Features']
    all_info = all_info.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
    
    print("The total time for searching subset: {}".format(dt.now()-st_t))
    
    return all_info, all_model, f

def xgboost_forward(X_train, y_train, X_test, y_test, n_selected_features = 1000, scoring='accuracy'):
    from sklearn.model_selection import TimeSeriesSplit
    from datetime import datetime as dt
    import random
    import warnings 
    
    warnings.filterwarnings("ignore")
    
    st_t = dt.now()
    
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
    
    cv_timeSeries = TimeSeriesSplit(n_splits=5).split(X_train)
    xgb = XGBClassifier(learning_rate=0.02, objective='multi:softmax', silent=True, nthread=20)
    n_iter_search = 30
    scoring = scoring
    
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
                rsearch_cv = RandomizedSearchCV(estimator=xgb,
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
    
    print("The total time for searching subset: {}".format(dt.now()-st_t))
    
    return all_info, all_model, f


def evaluate_multiclass(best_clf, X_train, y_train, X_test, y_test, 
                        model="Random Forest", num_class=3, top_features=2, n_selected_features=2, class_name = ""):
    print("-"*100)
    print("~~~~~~~~~~~~~~~~~~ PERFORMANCE EVALUATION ~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Detailed report for the {} algorithm".format(model))
    
    best_clf.fit(X_train, y_train)
    y_pred = best_clf.predict(X_test)
    y_pred_prob = best_clf.predict_proba(X_test)
    
    test_accuracy = accuracy_score(y_test, y_pred, normalize=True) * 100
    points = accuracy_score(y_test, y_pred, normalize=False)
    print("The number of accurate predictions out of {} data points on unseen data is {}".format(
            X_test.shape[0], points))
    print("Accuracy of the {} model on unseen data is {}".format(
            model, np.round(test_accuracy, 2)))
    
    print("Precision of the {} model on unseen data is {}".format(
            model, np.round(metrics.precision_score(y_test, y_pred, average="macro"), 4)))
    print("Recall of the {} model on unseen data is {}".format(
           model, np.round(metrics.recall_score(y_test, y_pred, average="macro"), 4)))
    print("F1 score of the {} model on unseen data is {}".format(
            model, np.round(metrics.f1_score(y_test, y_pred, average="macro"), 4)))
    
    print("\nClassification report for {} model: \n".format(model))
    print(metrics.classification_report(y_test, y_pred))
    
    plt.figure(figsize=(12,12))
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    cnf_matrix_norm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    print("\nThe Confusion Matrix: \n")
    print(cnf_matrix)
    
    #class_name = ["CDI", "ignore-nonCDI", "Health"]
    # class_name = ["CRC", "Adenomas", "Health"]
    class_name = class_name
    cmap = plt.cm.Blues
    plt.imshow(cnf_matrix_norm, interpolation="nearest", cmap=cmap)
    plt.colorbar()
    fmt = ".2g"
    thresh = cnf_matrix_norm.max()/2
    for i, j in itertools.product(range(cnf_matrix_norm.shape[0]), range(cnf_matrix_norm.shape[1])):
        plt.text(j,i,format(cnf_matrix_norm[i,j], fmt), ha="center", va="center", 
                 color="white" if cnf_matrix_norm[i,j] > thresh else "black", fontsize=35)
    
    plt.xticks(np.arange(num_class), labels = class_name, fontsize=30)
    plt.yticks(np.arange(num_class), labels = class_name, fontsize=30)
    
    
    plt.ylabel("True label", fontsize=30)
    plt.xlabel("Predicted label", fontsize=30)
    plt.ylim((num_class - 0.5, -0.5))
    plt.show()
    
    print("\nROC curve and AUC")
    y_pred = best_clf.predict(X_test)
    y_pred_prob = best_clf.predict_proba(X_test)
    y_test_cat = np.array(pd.get_dummies(y_test))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_class):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_test_cat[:,i], y_pred_prob[:,i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_class)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_class):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        
    mean_tpr /= num_class
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(12,12))
    plt.plot(fpr["macro"], tpr["macro"], 
         label = "macro-average ROC curve with AUC = {} - Accuracy = {}%".format(
                 round(roc_auc["macro"], 2), round(test_accuracy, 2)),
         color = "navy", linestyle=":", linewidth=4)
    #colors = cycle(["red", "orange", "blue", "pink", "green"])
    colors = sns.color_palette()
    for i, color in zip(range(num_class), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label = "ROC curve of class {0} (AUC = {1:0.2f})".format(i, roc_auc[i]))   
    plt.plot([0,1], [0,1], "k--", lw=3, color='red')
    plt.title("ROC-AUC for {} with {} selected features".format(model, n_selected_features), fontsize=20)
    plt.xlabel("False Positive Rate", fontsize=15)
    plt.ylabel("True Positive Rate", fontsize=15)
    plt.legend(loc="lower right")
    plt.show()
    
    
    if model == "Random Forest" or model == "XGBoost":
        importances = best_clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        feature_tab = pd.DataFrame({"Features": list(X_train.columns),
                                    "Importance": importances})
        feature_tab = feature_tab.sort_values("Importance", ascending = False).reset_index(drop=True)
        
        index = feature_tab["Features"].iloc[:top_features]
        importance_desc = feature_tab["Importance"].iloc[:top_features]  
        feature_space = []
        for i in range(indices.shape[0]-1, -1, -1):
            feature_space.append(X_train.columns[indices[i]])
        
        fig, ax = plt.subplots(figsize=(20,20))
        ax = plt.gca()
        plt.title("Feature importances for {} selected features".format(n_selected_features), fontsize=30)
        plt.barh(index, importance_desc, align="center", color="blue", alpha=0.6)
        plt.grid(axis="x", color="white", linestyle="-")
        plt.xlabel("The Average of Decrease in Impurity", fontsize=20)
        plt.ylabel("Features", fontsize=30)
        plt.yticks(fontsize=30)
        plt.xticks(fontsize=20)
        ax.tick_params(axis="both", which="both", length=0)
        plt.show()
        
        return {"importance": feature_tab, 
            "y_pred": y_pred,
            "y_pred_prob": y_pred_prob}
    
    return {"y_pred": y_pred,
            "y_pred_prob": y_pred_prob}












































    