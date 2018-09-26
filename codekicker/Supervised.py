# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 11:36:27 2018
@author: Prashant Bhat

Task - Classifying tickets using supervised learning 
"""
from Preprocess import Vectorizer
from DataSet import RawData
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

class SupervisedLearning:
    
    def __init__(self):
        ''' Empty init'''
        pass
        
    def main(self):
        ''' Main method to kick start supervised learning
        
            Parameters
            ----------
         
            Returns
            -------
            dataset : pandas dataframe
            y_true : Original class / cluster labels
            y_pred : predicted class / cluster labels
            cm : confusion matrix 
            precision : precision score 
            recall : recall score
            f_1_score : F1 score
        
        '''
        dataset, cluster_size = RawData().getDataset()
        vectorizer = Vectorizer()
        X, y_true = vectorizer.vectorize(dataset)
        y_pred = self.randomForest(X, y_true)
        cm = self.confusionMatrix(y_pred, y_true)
        precision = precision_score(y_true, y_pred, average=None)
        recall = recall_score(y_true, y_pred, average=None)
        f_1_score = f1_score(y_true, y_pred, average=None)
        return dataset, y_true, y_pred, cm, precision, recall, f_1_score
        
    def randomForest(self, X, y_true):
        ''' Prediction using RandomForest Classifier
        
            Parameters
            ----------
            X : sparse matrix, [n_samples, n_features] Tf-idf-weighted document-term matrix.
            y_true : Dependent feature representing original class / cluster of each row.
         
            Returns
            -------
            y_pred : prediction labels for a given training set.
    
    
        '''
        classifier = RandomForestClassifier(n_estimators=500)
        kf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None)
        for train_index, test_index in kf.split(X):
          X_train, X_test = X[train_index], X[test_index] 
          y_train, y_test = y_true[train_index], y_true[test_index]
          classifier.fit( X_train, y_train)
          classifier.predict(X_test)
        y_pred = classifier.predict(X)
        classifier.score(X, y_true)
        return y_pred

    def confusionMatrix(self, y_pred, y_true):
        ''' create confusion matrix
        
            Parameters
            ----------
            y_pred : prediction labels for a given training set.
            y_true : Dependent feature representing original class / cluster of each row.
         
            Returns
            -------
            
        
        '''
        cm = confusion_matrix(y_true, y_pred)
        return cm

supervised_method = SupervisedLearning()
dataset, y_true, y_pred, cm, precision, recall, f_1_score = supervised_method.main()
