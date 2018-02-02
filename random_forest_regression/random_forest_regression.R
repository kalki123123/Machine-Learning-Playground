# author - Prashant Bhat
# bprash123@gmail.com


# random forest regression

# import dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# random forest regression 
install.packages('randomForest')
library('randomForest')

x_grid = dataset$Level

randomForestRegressor = randomForest(dataset$Salary ~ x_grid, data = dataset, ntree = 300)

#predict the values
y_pred = predict(randomForestRegressor, data.frame(x_grid))

# plot the results
# install.packages('ggplot2')
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(randomForestRegressor, newdata = data.frame(Level = x_grid))),colour = 'blue') +
  ggtitle('Truth or Bluff (Random Forest Regression)') +
  xlab('Level') +
  ylab('Salary')