#author - Prashant Bhat


#import dataset 
#index starts with '1'
dataset = read.csv('Salary_Data.csv')

#split dataset into train and test
#install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
#if true , its in training set.. 0.8 percent of entries will be in training set
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#fitting data to regressor - liner model 
regressor = lm(formula = Salary ~ YearsExperience,
               data = training_set)

#predicting test results 
y_pred = predict(regressor, newdata = test_set)

#visualize the results 
#install.packages('ggplot2')
library(ggplot2)
ggplot()+
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary), 
             colour = 'red')+
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue')+
  ggtitle('Salary vs Exerience (Training set)')+
  xlab('Years of Experience')+
  ylab('Salary')

ggplot()+
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary), 
             colour = 'red')+
  geom_line(aes(x = test_set$YearsExperience, y = predict(regressor, newdata = test_set)),
            colour = 'green')+
  ggtitle('Salary vs Exerience (Test set)')+
  xlab('Years of Experience')+
  ylab('Salary')











