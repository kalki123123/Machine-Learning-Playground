# author - prashant bhat
# SVM classification 

# import the dataset
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[, 3:5]

# splitting into test and train set 
library('caTools')
split = sample.split(dataset, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#scale your independent variables if necessary 
training_set[-3] = scale(training_set[-3])
test_set[-3] = scale(test_set[-3])

#svm classifier 
#install.packages('e1071')
library('e1071')
svmclassifier  = svm(formula = Purchased ~ .,
                data = training_set,
                type = 'C-classification',
                kernel = 'radial',
                gamma = 0.7)

y_predict = predict(svmclassifier, test_set)

# confusion matrix
cm = table(test_set[, 3], y_predict)