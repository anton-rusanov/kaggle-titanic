library(party)
library(pROC)
source('formulas.r')
source('commons.r')

## Trains the Conditional Inference Forest model and predicts 'Survived' for the test set.
predict_with_conditional_inference_forest <- function(train, test, formula, suffix) {
  print(paste('Starting building model CI -', suffix))
  cond_inf_forest <- cforest(
      formula,
      data = train,
      controls=cforest_unbiased(ntree=1000, mtry=10, trace=TRUE))

#  print('Standard importance')
#  system.time(std_importance <- varimp(cond_inf_forest, conditional = FALSE))
#  print(std_importance)
#  print('Conditional importance')
#  system.time(cond_importance <- varimp(cond_inf_forest, conditional = TRUE))
#  cond_importance

  print('Starting CI prediction')
  predictionList <- predict(cond_inf_forest, test, OOB=TRUE, type = 'prob')
  predictionN <- sapply(predictionList, simplify = 'vector', FUN = function(x) (x[1]))
  predictionY <- sapply(predictionList, simplify = 'vector', FUN = function(x) (x[2]))
  data.frame(Y = predictionY, N = predictionN)
}


## Builds model configuration for cross-validation of method 'cforest'.
## In contrast to caret's implementation of this, this method builds a configuration for two
## parameters: mtry and mincriterion.
build_cforest_model_config <- function(mtryValues = NULL, mincriterionValues = NULL) {
  parameters <- data.frame(
      parameter = c('mtry', 'mincriterion'),
      class = c('numeric', 'numeric'),
      label = c('sample vars per node (mtry)', 'Split value (mincriterion)'))

  grid <- function(x, y, len = NULL) {
    if (is.null(mtryValues)) {
      mtrys <- c(5, 7, 9)
    } else {
      mtrys <- mtryValues
    }

    if (is.null(mincriterionValues)) {
      mincriterions <- c(qnorm(0.5))
    } else {
      mincriterions <- mincriterionValues
    }
    expand.grid(mtry = mtrys, mincriterion = mincriterions)
  }

  fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
    if (is.data.frame(x)) {
      data <- x
    } else {
      data <- as.data.frame(x)
    }
    data$.outcome <- y

    cforest(
        formula = as.formula(.outcome ~ .),
        data = data,
        # Cannot use cforest_unbiased, since that one does not allow overrding mincriterion.
        controls=cforest_control(
            ntree=1000,
            mtry=param$mtry,
            mincriterion=param$mincriterion,
            teststat = "quad",
            testtype = "Univariate",
            savesplitstats = FALSE,
            replace = FALSE,
            fraction = 0.632,
            trace = FALSE,
            ...))
  }

  # Get the caret's implementation of CV model info and replace parameters, grid and fit function.
  modelInfo <- getModelInfo('cforest')

  modelConfig <- list(
      parameters = parameters,
      grid = grid,
      fit = fit,
      type = modelInfo$cforest$type,
      library = modelInfo$cforest$library,
      loop = modelInfo$cforest$loop,
      predict = modelInfo$cforest$predict,
      prob = modelInfo$cforest$prob,
      sort = modelInfo$cforest$sort,
      levels = modelInfo$cforest$levels,
      varImp = modelInfo$cforest$varImp,
      label = modelInfo$cforest$label
  )
}


## Cross-validates  conditional inference RF using caret and prints the best model parameters.
## Returns the prediction for the validation set.
cross_validate_cforest <- function(partition, formula) {
  modelConfig <- build_cforest_model_config()
  trainedModel <-
      train_with_caret(modelConfig, partition$trainingSet, formula, 'cforest')
  print(trainedModel, digits = 3)
  plot(trainedModel,  scales = list(x = list(log = 2)))
  predict_with_model(trainedModel, partition$validationSet, 'cforest')
}


predict_with_cforest <- function(training, test, formula, label) {
  modelConfig <- build_cforest_model_config(mtry = 10, mincriterion = 0)
  trainedModel <- train_with_caret(modelConfig, training, formula, 'cforest',
      trainConfig = list(
          method = 'repeatedcv', repeats = 3, tuneLength = 16, verbose = FALSE))
  print(trainedModel, digits = 3)
  predict_with_model(trainedModel, test, 'cforest')
}
