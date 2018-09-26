# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 15:43:32 2018
@author: Prashant Bhat

Task - Tf-idf vectorization   
"""
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
class Vectorizer:
    
    def __init__(self):
        ''' Empty init'''
        pass
    
    def vectorize(self, dataset):
        ''' Generates tf-idf vectors for a given dataset.  
        
            Parameters
            ----------
            dataset - pandas dataframe
         
            Returns
            -------
            X : sparse matrix, [n_samples, n_features] Tf-idf-weighted document-term matrix.
            y : Dependent feature representing class / cluster of each row.
        
        '''
        corpus = self.generateCorpus(dataset)
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer()
        X = tfidf.fit_transform(corpus)
        y = dataset['y'].values
        return X, y
    
    def generateCorpus(self, dataset):
        ''' Preprocesses given dataset and returns Bag of features. .  
        
            Parameters
            ----------
            dataset - pandas dataframe
         
            Returns
            -------
            corpus : 2D list of features with each row correspoding to one entry in dataset.
        
        '''
        corpus = []
        ps = nltk.stem.PorterStemmer()
        for i in range(0, len(dataset.index)):
            ticket = re.sub(u'[^a-zA-Z0-9üöäß]', ' ', dataset['ticket'][i])
            ticket = ticket.lower()
            ticket = ticket.split()
            ticket = [ps.stem(word) for word in ticket if word not in set(stopwords.words('german'))]
            ticket = ' '.join(ticket)
            corpus.append(ticket)
        return corpus