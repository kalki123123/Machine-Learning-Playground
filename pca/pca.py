# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 10:38:24 2018
@author: Prashant Bhat

Principle Component Analysis (PCA)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Wine.csv')
X = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values


#test train split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Dimentionality Reduction - PCA 
# First set n_components to None, check  pca.explained_variance_ratio_ to decide how many variables you want
# Then, set the same number to n_components.

#PCA doesnot depend on dependent varialbe - hence the name - unsupervised learning 
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_


# logistic regression
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# predict values 
y_predict = log_reg.predict(X_test)

# confusion matrix -  to see how much error we committed while predicting on test set
from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(y_test, y_predict)
