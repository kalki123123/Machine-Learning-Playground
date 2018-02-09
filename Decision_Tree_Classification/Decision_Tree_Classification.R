#author - prashant bhat
# decision tree classification 

# import the dataset 
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[, 3:5]

# Encoding the target feature as factor
dataset$Purchased = factor(dataset$Purchased, levels = c(0, 1))

# feature scaling the dataset 
dataset[, -3] = scale(dataset[, -3])

# split the dataset into test set and training set 
library('caTools')
split = sample.split(dataset, SplitRatio = 3/4)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

library(rpart)
classifier = rpart(formula = Purchased ~ .,
                   data = training_set,
                   method = 'class')

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-3], type = 'class')

# confusion matrix
cm = table(test_set[, 3], y_pred)


# Note : 
# Donot forget to factorize your dependent variable since algorithm does not recognize this inherently. 






