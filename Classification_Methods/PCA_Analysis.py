#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 22:28:17 2019

@author: tungutokyo
"""

import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None
pd.set_option("display.max_columns", 60)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import decomposition
from datetime import datetime as dt

def standardize(data):
    from sklearn.preprocessing import StandardScaler
    data_std = StandardScaler().fit_transform(data)
    return data_std

def pca_vis(data, labels):
    st = dt.now()
    pca = decomposition.PCA(n_components=6)
    pca_reduced = pca.fit_transform(data)
    
    print("The shape of transformed data", pca_reduced.shape)
    print(pca_reduced[0:6])
    
    pca_data = np.vstack((pca_reduced.T, labels)).T
    print("The shape of data with labels", pca_data.shape)
    
    pca_df = pd.DataFrame(data=pca_data, columns=("1st_principal_component", "2nd_principal_component",
                                                  "3rd_principal_component", "4th_principal_component", 
                                                  "5th_principal_component", "6th_principal_component", "labels"))
    print(pca_df.head())
    
    sns.FacetGrid(pca_df, hue="labels", height=8).map(sns.scatterplot, 
            "1st_principal_component", "2nd_principal_component", edgecolor="w").add_legend()
    plt.xlabel("1st_principal_component ({}%)".format(round(pca.explained_variance_ratio_[0]*100),2), fontsize=15)
    plt.ylabel("2nd_principal_component ({}%)".format(round(pca.explained_variance_ratio_[1]*100),2), fontsize=15)
    
    sns.FacetGrid(pca_df, hue="labels", height=8).map(sns.scatterplot, 
            "3rd_principal_component", "4th_principal_component", edgecolor="w").add_legend()
    plt.xlabel("3rd_principal_component ({}%)".format(round(pca.explained_variance_ratio_[2]*100),2), fontsize=15)
    plt.ylabel("4th_principal_component ({}%)".format(round(pca.explained_variance_ratio_[3]*100),2), fontsize=15)
    
    sns.FacetGrid(pca_df, hue="labels", height=8).map(sns.scatterplot,
            "5th_principal_component", "6th_principal_component", edgecolor="w").add_legend()
    plt.xlabel("5th_principal_component ({}%)".format(round(pca.explained_variance_ratio_[4]*100),2), fontsize=15)
    plt.ylabel("6th_principal_component ({}%)".format(round(pca.explained_variance_ratio_[5]*100),2), fontsize=15)
    plt.show()
    
    print("Time taken to perform Principal Component Analysis: {}".format(dt.now()-st))
    return pca_df
    
def pca_redu(data, num_components):
    pca = decomposition.PCA(n_components=num_components)
    pca_data = pca.fit_transform(data)
    plt.figure(figsize=(15,10))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.axis('tight')
    plt.grid()
    plt.axhline(0.95, c='r')
    plt.xlabel("Number of components", fontsize=15)
    plt.ylabel("Cumulative explained variance", fontsize=15)
    plt.legend()
    return pca_data