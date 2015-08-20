library(RCurl)

## Writes the given solution to a file.
write_solution <- function(prediction, label, passengerIds, threshold = 0.5) {
  prediction01 <- ifelse(prediction$Y >= threshold, 1, 0)
  solution <- data.frame(PassengerId = passengerIds, Survived = prediction01)
  write.csv(solution, file=paste0('titanic-', label, '.csv'), row.names=FALSE, quote=FALSE)
  return (solution)
}


# Trains the given model (method + formula) on the training dataset, using caret library.
train_with_caret <-
    function(method, train, formula, label, userTrainConfig = list(), ...) {
  print(paste('Starting building model', label))
  defaultTrainConfig <- list(method = 'repeatedcv', repeats = 1, tuneLength = 8, verbose = FALSE)
  trainConfig <- merge.list(userTrainConfig, defaultTrainConfig)

  fitControl <- trainControl(method = trainConfig$method,
      number = 10,
      repeats = trainConfig$repeats,
      classProbs = TRUE,
      verbose = trainConfig$verbose)

  # For 'ROC' training metric, use summaryFunction = twoClassSummary in trainControl(..).
  fit <- train(formula, data = train,
    method = method,
    trControl = fitControl,
    preProc = c('BoxCox', 'center', 'scale'),
    tuneLength = trainConfig$tuneLength,
    metric = 'Accuracy',
    ...)
}


# Given a trained model, predicts the unknowns of the test set.
predict_with_model <- function(trainedModel, test, label) {
  print(paste('Starting prediction for', label))
  predict(trainedModel, test, type = 'prob')
}


# Trains and predicts the given model (method + formula) on the training dataset and predicts the
# unknowns of the test set.
train_and_predict_with_caret <-
    function(method, training, test, formula, label, userTrainConfig = list(), ...) {
  trainedModel <- train_with_caret(method, training, formula, label, userTrainConfig, ...)
  predict_with_model(trainedModel, test, label)
}


