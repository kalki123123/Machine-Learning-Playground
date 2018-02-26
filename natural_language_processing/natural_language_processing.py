# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 09:46:44 2018
@author: Prashant Bhat

Natural language Processing
BoW = Bag of Words model + Naive bayes Classifier 
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import the dataset 
# ususally use tab as a delimiter , since your data might also contain 
# comma, its way better to use tabs as a delimiter 
# use *.tsv file 
dataset = pd.read_csv('Restaurant_Reviews.tsv', sep='\t', quoting=3)

# BoW = Bag of Words model 
# clean your dataset 
# remove all separators, lower case your review,
# remove stop words and stemming  
import re
import nltk 
#nltk.download('stopwords')
from nltk.corpus import stopwords
ps = nltk.stem.PorterStemmer()
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# text cleaning can also be done here
# create bag of words - sparse matrix 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features= 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# test train split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y )

# Naive bayes classifier 
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# predict the values for test set 
y_pred = classifier.predict(X_test)

# confusion matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

