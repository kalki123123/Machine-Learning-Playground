# author - prashant bhat
# KNN classification 

#import the library 
install.packages('class')
library('class')

# import the data
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[,3:5]

# split training and test set 
library('caTools')
splittingRatio = sample.split(dataset, SplitRatio = 0.75)
training_set = subset(dataset, splittingRatio == TRUE)
test_set = subset(dataset, splittingRatio == FALSE)


# kNN classification 
# cl - factor fo true classification - its the dependent variable 
kNearest = knn(train = training_set[, -3], test =  test_set[, -3], cl = training_set[, 3], k = 5 )
summary(kNearest)

#confusion matrix 
confusion_matrix = table(test_set[, 3], kNearest)