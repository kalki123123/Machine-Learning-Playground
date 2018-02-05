# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:32:09 2018
@author: Prashant bhat

K-Nearest Neighbour Classification (KNN)
"""

# import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import dataset 
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, 1:4].values
y = dataset.iloc[:, 4].values

# label encoding and OneHotEncoder for 'Gender column in the dataset'
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncode = LabelEncoder()
X[:, 0] = labelEncode.fit_transform(X[:, 0]) 
oneHotEncode = OneHotEncoder(categorical_features=[0])
X = oneHotEncode.fit_transform(X).toarray()

#get rid of dummy variable trap 
X = X[:, 1:4]

#test train split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier 
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

# predict the results 
y_predict = neigh.predict(X_test)

#confusion matrix to analyze prediction 
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_predict)






















