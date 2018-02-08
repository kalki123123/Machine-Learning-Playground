# author - Prashant bhat
# Naive Bayes classifier 

# import the dataset 
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[, 3:5]

# naive bayes in e1071 does not recognize our dependent varialbe as factors
# Therefore factorize the dependent variable. 
dataset$Purchased = factor(dataset$Purchased, levels = c(0, 1))

# test train split 
library('caTools')
split = sample.split(dataset, SplitRatio = 3/4)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# feature scaling - necessary in this dataset 
training_set[, -3] = scale(training_set[, -3])
test_set[, -3] = scale(test_set[, -3])

# naive bayes classifier 
library('e1071')
classifier = naiveBayes(x = training_set[-3],
                        y = training_set$Purchased)

# predict the values for training set 
y_pred = predict(classifier, newdata = test_set[, -3])

# confusion matrix 
cm = table(test_set[, 3], y_pred)

# Special Note :
# there is a classical error in machine learning in R
# if you get y_pred = factor(0), Levels: 
# Plese remember that your classifier algorithm is not recognizing your dependent variable as 
# factorized categorical features 
#
# Please factorize the dependent variable in this case !!








