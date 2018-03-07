# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 14:21:34 2018
@author: Prashant Bhat

Artificial Neural Network
( binary classification problem )
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------- Preprocessing ------------------------------
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# label encoding 
# DONOT forget dummy variable trap !!
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_1 = LabelEncoder()
X[:, 1] = labelencoder_1.fit_transform(X[:, 1])
labelencoder_2 = LabelEncoder()
X[:, 2] = labelencoder_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]


# split the dataset 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Scale the independent variables
from sklearn.preprocessing import StandardScaler
standardscalar = StandardScaler()
X_train = standardscalar.fit_transform(X_train)
X_test = standardscalar.transform(X_test)



#  --------------- ANN -------------------------------------------
import keras
from keras.models import Sequential
model = Sequential()

from keras.layers import Dense
model.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
model.add(Dense(units=6,  kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=1,  kernel_initializer='uniform', activation='linear'))

# stochastic gradient descent - keras compile 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# fit your model and predict 
model.fit(X_train, y_train, epochs=100, batch_size=10)

loss_and_metrics = model.evaluate(X_test, y_test, batch_size=100)

y_pred = model.predict(X_test, batch_size=50)
y_pred = y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)









