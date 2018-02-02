# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 15:06:20 2018
@author: Prashant Bhat

random forest regression 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import dataset 
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

# random forest regression 
# n_estimators = number of decision trees to arrive at the final decision 
from sklearn.ensemble import RandomForestRegressor
randomForest = RandomForestRegressor(n_estimators=300)
randomForest.fit(X, Y)

# predict new value 
y_pred = randomForest.predict(6.5)

# plot the results 
X_grid = np.arange(0, 12, 0.1)[:, np.newaxis]
plt.scatter(X, Y, color = 'blue')
plt.plot(X_grid, randomForest.predict(X_grid), color = 'red')
plt.show()



