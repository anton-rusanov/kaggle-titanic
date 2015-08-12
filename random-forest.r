library(caret)
library(inTrees)
library(plyr)
library(randomForest)


## Trains the Random Forest model and predicts 'Survived' for the test set.
predict_with_random_forest <- function(train, test, formula, suffix) {
  print(paste('Starting building model RF -', suffix))

  my_forest <- randomForest(
      formula,
      train,
      importance = TRUE,
      proximity = TRUE,
      ntree = 5000,
      mtry = 3)
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

build_rf_model_config <- function(mtryValues = NULL) {
  # Get the caret's implementation of CV model info and replace grid function.
  modelInfo <- getModelInfo('rf', regex = FALSE)

  grid <- function(x, y, len = NULL) {
    if (is.null(mtryValues)) {
      mtrys <- caret::var_seq(p = ncol(x), classification = is.factor(y), len = len)
    } else {
      mtrys <- mtryValues
    }
    data.frame(mtry = mtrys)
  }

  modelConfig <- list(
      grid = modelInfo$rf$grid,
      fit = modelInfo$rf$fit,
      parameters = modelInfo$rf$parameters,
      type = modelInfo$rf$type,
      library = modelInfo$rf$library,
      loop = modelInfo$rf$loop,
      predict = modelInfo$rf$predict,
      prob = modelInfo$rf$prob,
      sort = modelInfo$rf$sort,
      levels = modelInfo$rf$levels,
      varImp = modelInfo$rf$varImp,
      label = modelInfo$rf$label
  )
}


## Cross-validates RF using caret and prints the best model parameters.
## Returns the prediction for the validation set.
cross_validate_rf <- function(partition, formula) {
  modelConfig <- build_rf_model_config()
  trainedModel <-
      train_with_caret(modelConfig, partition$trainingSet, formula, 'rf',
      trainConfig = list(
                method = 'repeatedcv', repeats = 1, tuneLength = 16, verbose = TRUE))
  print(trainedModel, digits = 3)
  plot(trainedModel,  scales = list(x = list(log = 2)))
  predict_with_model(trainedModel, partition$validationSet, 'rf')
}


predict_with_rf <- function(training, test, formula, label) {
  modelConfig <- build_rf_model_config(mtryValues = c(4))
  trainedModel <- train_with_caret(modelConfig, training, formula, 'rf',
      trainConfig = list(
          method = 'repeatedcv', repeats = 3, tuneLength = 16, verbose = FALSE))
  print(trainedModel, digits = 3)
  predict_with_model(trainedModel, test, 'rf')
}
