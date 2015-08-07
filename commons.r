## Writes the given solution to a file.
write_solution <- function(prediction, label, passengerIds, threshold = 0.5) {
  prediction01 <- ifelse(prediction$Y >= threshold, 1, 0)
  solution <- data.frame(PassengerId = passengerIds, Survived = prediction01)
  write.csv(solution, file=paste0('titanic-', label, '.csv'), row.names=FALSE, quote=FALSE)
  return (solution)
}


# Trains the given model (method + formula) on the training dataset, using caret library.
train_with_caret <-
    function(method, train, formula, label,
        trainConfig = list(repeats = 3, tuneLength = 16, verbose = FALSE)) {
  print(paste('Starting building model', label))
  fitControl <- trainControl(method = 'repeatedcv',
      number = 10,
      repeats = trainConfig$repeats,
      classProbs = TRUE,
      summaryFunction = twoClassSummary)

  fit <- train(formula, data = train,
    method = method,
    trControl = fitControl,
    preProc = c('center', 'scale'),
    tuneLength = trainConfig$tuneLength,
    metric = 'ROC',
    prob.model = TRUE,
    verbose = trainConfig$verbose)
}


# Given a trained model, predicts the unknowns of the test set.
# TODO: extract writing from here and get rid of the flag
predict_with_model <- function(trainedModel, test, label) {
  print(paste('Starting prediction for', label))
  predict(trainedModel, test, type = 'prob')
}

# Trains and predicts the given model (method + formula) on the training dataset and predicts the
# unknowns of the test set.
train_and_predict_with_caret <-
    function(method, training, test, formula, label, threshold = 0.5) {
  trainedModel <- train_with_caret(method, training, formula, label)
  predict_with_model(trainedModel, test, label)
}


