#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 15:37:14 2020

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
class_name = ["CDI", "ignore-nonCDI", "Health"]
X_train, X_test, y_train, y_test = train_test_split(microbiome, y, test_size=0.3, random_state=42)

###############################################################################
################### Step 2 - Run selected models ##############################
###############################################################################

automl = AutoML_classification()

rf_best, _, _, _, _ = automl.Random_Forest(X_train, y_train, X_test, y_test)
evaluate_rf = automl.evaluate_multiclass(rf_best, X_train, y_train, X_test, y_test,
                            model = "Random Forest", num_class=3, top_features=20, class_name = class_name)


###############################################################################
####################### Random Intersection Tree ##############################
###############################################################################

from sklearn import tree
import graphviz 
import pydot
from subprocess import call
from sklearn.tree import _tree

rf_best = best_model_119
rf_best.fit(X_train, y_train)
dtree = rf_best.estimators_[0]
feature_names = list(X_train.columns)


tree.plot_tree(dtree)
r = tree.export_text(dtree, feature_names=feature_names)

dot_data = tree.export_graphviz(dtree, feature_names = X_train.columns, out_file="CDI.dot")
(graph, ) = pydot.graph_from_dot_file('CDI.dot')
graph.write_png('CDI.png')

call(['dot', '-Tpng', 'CDI.dot', '-o', 'CDI.png', '-Gdpi=600'])

def all_tree_signed_paths(dtree, root_node_id = 0):

    children_left = dtree.tree_.children_left
    children_right = dtree.tree_.children_right

    #root_node_id = 0
    if root_node_id is None:
        paths = []
    
    feature_id = dtree.tree_.feature[root_node_id]
    if children_left[root_node_id] != _tree.TREE_LEAF:
        paths_left = [[(root_node_id, 'L')] + l 
                  for l in all_tree_signed_paths(dtree, children_left[root_node_id])]
        paths_right = [[(root_node_id, 'R')] + l
                  for l in all_tree_signed_paths(dtree, children_right[root_node_id])]
        paths = paths_left + paths_right
    else:
        paths = [[(root_node_id, )]]
        
    return paths

paths = all_tree_signed_paths(dtree = dtree)

###############################################################################
def compute_impurity_decrease(dtree):
    
    impurity = dtree.tree_.impurity
    weight = dtree.tree_.n_node_samples 
    weight = [x / weight[0] for x in weight]
    impurity_decrease = []
    n_nodes = len(weight)
    for i in range(n_nodes):
        left_child = dtree.tree_.children_left[i]
        right_child = dtree.tree_.children_right[i]
        if left_child < 0 or right_child < 0:
            impurity_decrease.append(-1)
        else:
            curr_impurity = weight[i] * impurity[i]
            left_impurity = weight[left_child] * impurity[left_child]
            right_impurity = weight[right_child] * impurity[right_child]
            impurity_decrease.append(curr_impurity - left_impurity - right_impurity)
            
    return impurity_decrease

impurity_decrease = compute_impurity_decrease(dtree = dtree)

###############################################################################


def get_filtered_dt_feature_paths(dtree, threshold, signed=False,
                               weight_scheme="depth"):
    
    impurity_decrease = compute_impurity_decrease(dtree)
    features = dtree.tree_.feature
    features_names_ = [feature_names[i] for i in features]
    threshold = 0.01
    filtered = [x > threshold for x in impurity_decrease]
    tree_paths = all_tree_signed_paths(dtree=dtree)
    feature_paths = []
    for path in tree_paths:
        tmp = [(features_names_[x[0]], x[1]) for x in path if filtered[x[0]]]
        cleaned = []
        cache = set()
        for k in tmp:
            if k[0] not in cache:
                cleaned.append(k)
                cache.add(k[0])
        feature_paths.append(cleaned)
    if weight_scheme == "depth":
        weight = [2 ** (1-len(path)) for path in tree_paths]        
    elif weight_scheme == "samplesize":
        samplesize_per_node = dtree.tree_.weight_n_node_samples
        weight = [samplesize_per_node[path[-1]] for path in tree_paths]
    
    total = sum(weight)
    weight = [w / total for w in weight]
    
    return feature_paths, weight

###############################################################################

import pyfpgrowth
import Find_Freq_Pattern as ffp
from collections import OrderedDict
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from itertools import chain

feature_paths = [list(path) for path in feature_paths]
patterns = pyfpgrowth.find_frequent_patterns(feature_paths, 20)
# patterns = ffp.find_frequence_patterns(feature_paths, 7)
prevalence = {p:0 for p in patterns}
for key in patterns:
    p = set(list(key))
    for path, w in zip(feature_paths, weight):
        if p.issubset(path):
            prevalence[key] += w
            
prevalence = OrderedDict(sorted(prevalence.items(), key=lambda t: -t[1] ** (1/len(t[0]))),)
interaction_names = list(prevalence.keys())

df_prevalence = pd.DataFrame.from_dict(prevalence, orient='index', 
                                       columns=["Interaction_Score"]).sort_values(
                                           by="Interaction_Score", ascending=False)

df_prevalence["Order"] = [len(x) for x in df_prevalence.index]


df_prevalence = df_prevalence.reset_index().rename(columns={"index": "Interactions"})

index_order_1 = df_prevalence.index[df_prevalence["Order"] < 2]
df_prevalence.drop(index_order_1, inplace=True)
df_prevalence = df_prevalence.sort_values(by="Order", ascending=False)
index_clean = list(df_prevalence["Interactions"])

inter_clean = []
for k in index_clean: 
    k_sub = []
    for k_1 in k:
        k_sub.append(k_1[0])
        if k_1 is not k:
            k_sub.append("--")
        
    inter_clean.append(k_sub)
    

df_prevalence["Interactions"] = inter_clean 

df_prevalence_2 = df_prevalence[df_prevalence["Order"] == 2].sort_values(
                                by="Interaction_Score", ascending=False).reset_index(drop=True)
df_prevalence_3 = df_prevalence[df_prevalence["Order"] == 3].sort_values(
                                by="Interaction_Score", ascending=False).reset_index(drop=True)
df_prevalence_4 = df_prevalence[df_prevalence["Order"] == 4].sort_values(
                                by="Interaction_Score", ascending=False).reset_index(drop=True)
df_prevalence_5 = df_prevalence[df_prevalence["Order"] == 5].sort_values(
                                by="Interaction_Score", ascending=False).reset_index(drop=True)

df_prevalence_sub = pd.concat([df_prevalence_2.iloc[0:5], df_prevalence_3.iloc[0:5], 
                               df_prevalence_4.iloc[1:3], df_prevalence_5.iloc[0:1]], axis=0).reset_index(drop=True)

df_prevalence_sub["Interactions"] = ["".join(chain.from_iterable(x)) for x in df_prevalence_sub["Interactions"]]

plt.figure(figsize=(10,10))
ax = sns.stripplot(x="Interaction_Score", y = "Interactions", 
                   hue="Order", data=df_prevalence_sub, size=10)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.xlabel("Interaction Score", fontsize=20)
plt.ylabel("")

def visualize_prevalent_interactions(prevalence, **kwargs):
    
    #orders = [len(x) for x in prevalence]
    #log2_prevalence = [np.log(x) / np.log(2) for x in prevalence.values()]
    interaction_names = prevalence.keys()
    interaction_score = prevalence.values()
    plt.scatter(interaction_score, interaction_names, alpha=0.7)
    #plt.plot([0, max(orders)+0.5], [0, -max(orders)-0.5])
    #plt.xlim(0, max(orders)+0.5)
    #plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    #plt.ylim(min(log2_prevalence)-0.5,0)
    #plt.ylabel('log2(prevalence)')
    plt.xlim(-0.01, 0.6)
    plt.xlabel("Interaction Score")
    plt.ylabel('order of the interactions')
    plt.show()
    
visualize_prevalent_interactions(prevalence)    

















































