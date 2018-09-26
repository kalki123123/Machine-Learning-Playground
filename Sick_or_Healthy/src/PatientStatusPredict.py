# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 12:09:32 2018
@author: Prashant bhat

Task - Patient status prediction 
"""
# import independent features  
import pandas as pd
import numpy as np
features = pd.read_csv('../data/features.csv')
features.shape

# bivariate analysis
corr_matrix = features.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
features = features.drop(features[to_drop], axis=1)
features.shape

# univariate analysis 
X = features.iloc[:, 1:].values
from sklearn.feature_selection import VarianceThreshold
vt = VarianceThreshold(0.2)
X = vt.fit_transform(X)
X.shape

# import labels - 1-sick or 0-healthy
from matplotlib import pyplot as plt 
labels = pd.read_csv('../data/labels.csv')
y = labels.iloc[:, 1].values
x_1 = [0,1]
x_2 = [y.size - np.count_nonzero(y), np.count_nonzero(y)]
plt.figure(1)
plt.bar(x_1, x_2,width=0.15, align = 'center') 
plt.title('Distribution of original labels') 
plt.ylabel('Number of Patients') 
plt.xlabel('Sickness status')  
plt.show()

# feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.decomposition import PCA
pca = PCA(n_components = 50)
X = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_
print(sum(explained_variance))

# create ANN model 
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

classifier = Sequential()
classifier.add(Dropout(0.3, input_shape = (50,)))
classifier.add(Dense(21,  activation='relu', kernel_initializer = 'uniform'))
classifier.add(Dropout(0.5))
classifier.add(Dense(21, activation='relu', kernel_initializer='uniform'))
classifier.add(Dropout(0.5))
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='uniform'))

# compile, fit and predict 
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier.fit(X, y, batch_size=10, epochs=500)
y_pred = classifier.predict(X)
y_pred = (y_pred > 0.4)



# roc curve
from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
plt.figure(2)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.2,1.2])
plt.ylim([-0.2,1.2])
plt.ylabel(' True Positive Rate')
plt.xlabel(' False Positive Rate')
plt.show()


from sklearn.metrics import confusion_matrix, recall_score
import seaborn as sn
cm = confusion_matrix(y, y_pred)
plt.figure(figsize = (7,5))
plt.title('Confusion Matrix')
sn.heatmap(cm, annot=True, cmap='Greens', fmt='g')

# recall -
recall = recall_score(y, y_pred)
print(recall)







































