# -*- coding: utf-8 -*-
"""
Spyder Editor
author - Prashant Bhat
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('your_data_set', header = None)

transactions = []
for i in range (0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

 
#  apriori
from apyori import apriori
res = apriori(transactions, min_support = 0.003, min_confidence=0.2 , min_lift = 3, min_length = 2)
final_result = list(res)


# This function takes as argument your results list and return a tuple list with the format:
# [(rh, lh, support, confidence, lift)] 
def inspect(results):
    rh          = [tuple(result[2][0][0]) for result in results]
    lh          = [tuple(result[2][0][1]) for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(rh, lh, supports, confidences, lifts))
# this command creates a data frame to view
resultDataFrame=pd.DataFrame(inspect(final_result),
                columns=['rhs','lhs','support','confidence','lift'])
