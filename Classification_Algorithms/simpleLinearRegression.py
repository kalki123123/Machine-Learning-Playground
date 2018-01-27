# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 14:51:22 2017

@author: bpras
"""
#index in python start at 0 -----------------------------------------------
#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing data set
dataset = pd.read_csv('Salary_Data.csv')
#get all rows and columns except last column
X = dataset.iloc[:, :-1].values
#get only the third colum
Y = dataset.iloc[:, 1].values

#splitting the dataset into training and test sets
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train , Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

#Feature scaling not necessary here.. algorithm will take care of it here
#fitting the simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

#Predicting the test set results 
Y_pred = regressor.predict(X_test)

#visualize training and test set results 
plt.scatter(X_train, Y_train, color = 'red') # training points Xtrain, Ytrain
plt.plot(X_train, regressor.predict(X_train), color = 'blue') # regression line Xtrain, YtrainPrediction
plt.title('Salary Vs Experience (Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test, Y_test, color = 'green') # training points Xtrain, Ytrain
plt.plot(X_train, regressor.predict(X_train), color = 'blue') # regression line Xtrain, YtrainPrediction
# changing above line to Xtest vs YtestPrediction, results remains same.. because our regressor is aleady
#trained on the training set. So linear regression model remains the same
plt.title('Salary Vs Experience (Test set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()









