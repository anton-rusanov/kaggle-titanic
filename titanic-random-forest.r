library(caret)
library(randomForest)


## Trains the Random Forest model and predicts 'Survived' for the test set.
predict_with_random_forest <-
    function(train, test, formula, suffix, threshold = 0.5) {
  print(paste('Starting building model RF -', suffix))

  my_forest <- randomForest(
      formula,
      train,
      importance = TRUE,
      proximity = TRUE,
      ntree = 2000)
  # TODO Check nodesize=10, Setting this number larger causes smaller trees to be grown
  # TODO Check replace=TRUE, Should sampling of cases be done with or without replacement?
  print('Variable importance')
  print(round(importance(my_forest), 2))
#  varImpPlot(my_forest)
#  print('Variable proximity')
#  print(summary(my_forest$proximity))


  print('Starting RF prediction')
  # Make your prediction using the test set
  predictionMatrix <- predict(my_forest, test, type = 'prob')
  as.data.frame(predictionMatrix)
}
