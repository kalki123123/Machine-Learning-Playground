# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 11:22:14 2018

@author: prashant bhat
"""

# polynomial regression 

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_salaries.csv')

X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

# split in to test and train dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)

from sklearn.preprocessing import PolynomialFeatures
#choose degree based on trial and error - there is no standard value
polyFeatures = PolynomialFeatures(degree=4)
X_poly_train = polyFeatures.fit_transform(X_train, Y_train)
X_poly_test = polyFeatures.fit_transform(X_test, Y_test)

# predict the output for test and training set 
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_poly_train, Y_train)

y_train_predict = lin_reg.predict(X_poly_train)
y_test_predict = lin_reg.predict(X_poly_test)

# plot the results 
plt.scatter(X_test, Y_test, color = 'red')
plt.scatter(X_train, Y_train, color = 'blue')
plt.plot(X_train, y_train_predict, color = 'green', '-r')
plt.show()
