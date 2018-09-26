# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 13:59:46 2018

@author: Prashant Bhat
"""
# Aim - To predict wine quality given some alcohol composition

import pandas as pd
import numpy as np
import matplotlib as plt

# read data from csv file
dataset = pd.read_csv('winequality-white.csv', sep=';')
X = dataset.iloc[:, 0:12].values
Y = dataset.iloc[:, 11].values

# split test and training set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test  = train_test_split(X, Y, train_size=0.25)

# standard scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

'''
Before we dwell into classification task, it makes sense to come up with 
principle components i.e PCA

from sklearn.decomposition import PCA
pca = PCA(n_components=10)
X_train = pca.fit_transform(X_train)
print(pca.explained_variance_ratio_) 
X_test = pca.transform(X_test)

For this dataset, It seems PCA results are worse than not considering it at all !!
'''

# Naive bayes classifier 
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# predict the values for test set 
y_pred = classifier.predict(X_test)

# confusion matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
