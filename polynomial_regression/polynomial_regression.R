# author - Prashant bhat
# Polynomial regresssion 
dataset = read.csv('Position_Salaries.csv')

# fitting data to polynomial regression 
# use linear model along with polynomial function to fit your model
x_grid = dataset$Level
fit = lm(formula = dataset$Salary ~ poly(x_grid, degree = 5, raw = TRUE), data = dataset)
predict(fit, data.frame(x_grid = 7))

# Visualising the Regression Model results (for higher resolution and smoother curve)
# install.packages('ggplot2')
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(fit, newdata = data.frame(Level = x_grid))), colour = 'blue') +
  ggtitle('Truth or Bluff (Polynomial Regression)') +
  xlab('Level') +
  ylab('Salary')

# IMPORTANT NOTE 
# newdata names in lm() and predict should match. Otherwise you would get a warning 