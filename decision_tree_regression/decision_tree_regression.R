# author - prashant bhat

# Decision tree regression 
install.packages('rpart')
library('rpart')

#import data and remove not so useful features
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]
x = dataset$Level

#decision tree regression using rpart
# tweak minsplit as per your requirement 
decTree = rpart(dataset$Salary ~ x, data = dataset, minsplit = 2)
predict(decTree, data.frame(x=6.5))

# plot the results
x = seq(0, 12, 0.1)
ggplot()+
  geom_point(aes(x= dataset$Level, y = dataset$Salary), colour = 'red')+
  geom_line(aes(x = x, y = predict(decTree, data.frame(x))), colour = 'blue')+
  xlab('Employee position')+
  ylab('Salary')+
  ggtitle('Salary Prediction')

  

