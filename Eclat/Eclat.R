#author - Prashant Bhat

#import the dataset
dataset = read.csv('your_data_set', header = FALSE )

#eclat doesnot deals with transactions. Therefore change dataset to transactions
#rm.duplicates removes the duplicate entries in case of any error in data collection
install.packages('arules')
library('arules')
dataset = read.transactions('your_data_set',  sep = ',', rm.duplicates = FALSE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

#eclat algorithm 
# simple to train than apriori. However, certainly, not that efficient either !

rules = eclat(dataset, parameter = list(support = 0.003, minlen = 2))
inspect(sort(rules, by = 'support')[1:10])