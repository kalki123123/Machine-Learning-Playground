# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 14:28:41 2018
@author: Prashant Bhat

Decision Tree Classification 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import the dataset 
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, 2:4].values
y = dataset.iloc[:, 4].values

# scale your input 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# test train split 
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75)


# Decision Tree Classifier
from sklearn.tree  import DecisionTreeClassifier 
classifier = DecisionTreeClassifier( criterion='entropy')
classifier.fit(X_train, y_train)

# predict the output for test set 
y_predict = classifier.predict(X_test)

# confusion matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)

# visualize the results 
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree Classification (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()