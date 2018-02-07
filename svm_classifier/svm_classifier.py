# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 14:20:55 2018
@author: Prashant Bhat

Support Vector Machine classification 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import dataset 
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, 2:4].values
Y = dataset.iloc[:, 4].values

# scaling 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)


# split data into test and train set - 25% test size  by default
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

# svm classifier 
from sklearn.svm import SVC
svmClassifier = SVC()
svmClassifier.fit(X_train, Y_train)

#predict the results 
y_predict = svmClassifier.predict(X_test)

# confusion matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_predict, Y_test)