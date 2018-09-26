# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 15:43:32 2018
@author: Prashant Bhat

Task - Read complete dataset with true cluster labels and their cumulative size  
"""
import pandas as pd
from sklearn.utils import shuffle

class RawData:
    
    def __init__(self):
        ''' Empty init '''
        pass
    
    def getData(self, file):
        ''' Read data from given file. Data is read as tickets 
        
             Parameters
             ----------
             file : string reprenting name of file to be read.
             
             Returns
             -------
             data : pandas dataframe 
             
        '''
        data = pd.read_table(file, header=None, names=['ticket'])
        return data
    
    def getDataset(self):
        ''' Read each cluster_x.txt and concatenate.  
        
            Parameters
            ----------
         
         
            Returns
            -------
            data : pandas dataframe 
            cluster_size : list of integers representing cumulative size of clusters. Originally intended for KMeans
        
        '''
        clusters = []
        cluster_size = []
        for i in range(1, 6):
            cluster_data = self.getData('data/cluster_{}.txt'.format(i))
            y = [i-1 for j in range(0, len(cluster_data.index))]
            cluster_size.append(cluster_size[-1] + len(cluster_data.index)) if i!=1 else cluster_size.append(len(cluster_data.index))
            cluster_data['y'] = y
            clusters.append(cluster_data)
        dataset = pd.concat(clusters, ignore_index = True)
        dataset = shuffle(dataset)
        return dataset, cluster_size
    
        