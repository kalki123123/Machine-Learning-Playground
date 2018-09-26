# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 11:36:27 2018
@author: Prashant Bhat

Task - Clustering using KMeans 
"""

from sklearn.cluster import KMeans
from Preprocess import Vectorizer
from DataSet import RawData


class KMeansClustering:

    def __init__(self):
        ''' Empty init'''
        pass
    
    def clusterTickets(self, dataset):
        ''' Clustering using KMeans
        
            Parameters
            ----------
            dataset : pandas dataframe
         
            Returns
            -------
            y_pred : prediction labels for a given training set.
            y_true : Dependent feature representing original class / cluster of each row.
        '''
        vectorizer = Vectorizer()
        X, y_true = vectorizer.vectorize(dataset)
        kmeans = KMeans(n_clusters=5, init='k-means++', n_init=100, max_iter=1000)
        y_predict = kmeans.fit_predict(X)
        print(kmeans.get_params())
        return y_predict, y_true



kmeansClustering = KMeansClustering()        
dataset, cumulative_cluster_size = RawData().getDataset()
y_pred, y_true = kmeansClustering.clusterTickets(dataset)






















#for i in range(0, len(cumulative_cluster_size)):
#    if i > 0:
#        cluster_label = kmeansClustering.most_common(y_pred[cumulative_cluster_size[i-1]:cumulative_cluster_size[i]])
#        for j in range(cumulative_cluster_size[i-1], cumulative_cluster_size[i]):
#            y_true[j] = cluster_label[0][0]
#    else:
#        cluster_label = kmeansClustering.most_common(y_pred[0:cumulative_cluster_size[i]])
#        for j in range(0, cumulative_cluster_size[i]):
#            y_true[j] = cluster_label[0][0]















