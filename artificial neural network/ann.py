# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 17:24:54 2018

@author: psbhat
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
#remove dummy variable to skip dummy variable trap
X = X[:,1:]


# test train split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# feature scaling is must for deep learning since it involves lot of coomputation and 
#we dont want to influence one variable

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()
# input layer and first hidden layer
classifier.add(Dense(6,  activation='relu', kernel_initializer = 'uniform', input_shape = (11,)))

# more hidden layers
classifier.add(Dense(6, activation='relu', kernel_initializer='uniform'))

# output layer
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='uniform'))

#compiling ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# fit and predict the results
classifier.fit(X_train, y_train, batch_size=10, epochs=500)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# test a new entry - whether he leaves a bank
new_pred = classifier.predict(sc.transform(np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_pred = (new_pred > 0.5)


















