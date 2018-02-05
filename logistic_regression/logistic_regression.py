# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 11:19:04 2018
@author: Prashant Bhat 

Logistic Regression - classification task 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# logistic regression
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# predict values 
y_predict = log_reg.predict(X_test)

# confusion matrix -  to see how much error we committed while predicting on test set
from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(y_test, y_predict)





