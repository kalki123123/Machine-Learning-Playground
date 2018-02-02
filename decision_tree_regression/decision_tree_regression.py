# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 12:17:47 2018

@author: Prashant Bhat
"""

# Decision tree regression 
# CART - classification and regression tree technique

# import libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

from sklearn.tree import DecisionTreeRegressor
decisionTree = DecisionTreeRegressor(max_depth=3)
decisionTree.fit(X, Y)

#predict the particular output
y_pred = decisionTree.predict(X)

# plot the results 
# in order to clearly see decision tree results, plot across lot of points
X_grid = np.arange(0.0, 12,0.1)[:, np.newaxis]
y_predict = decisionTree.predict(X_grid)
plt.scatter(X, Y, color = 'blue')
plt.plot(X_grid, y_predict, color = 'red')
plt.xlabel('Position of employee')
plt.ylabel('Salary expectation ')
plt.title('Guess new employee salary')
plt.show()