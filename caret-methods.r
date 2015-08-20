library(gbm)
source('commons.r')

## Trains the Support Vector Machine model and predicts 'Survived' for the test set.
predict_with_caret_svm <- function(training, test, formula, label) {
  train_and_predict_with_caret('svmRadial', training, test, formula, label, prob.model = TRUE)
}


## Trains a Stochastic Gradient Boosting model and predicts 'Survived' for the test set.
predict_with_caret_gbm <- function(training, test, formula, label) {
  train_and_predict_with_caret('gbm', training, test, formula, label)
}

