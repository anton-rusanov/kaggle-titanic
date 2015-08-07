library(kernlab)
source('commons.r')


## Builds the caret model configuration for ksvm method.  
build_svm_laplacian_model_config <- function(sigmaValues = NULL, cValues = NULL) {
  parameters <- data.frame(parameter = c('C', 'sigma'),
      class = rep('numeric', 2),
      label = c('Cost', 'Sigma'))

  grid <- function(x, y, len = NULL) {
    if (is.null(sigmaValues)) {
      ## This produces low, middle and high values for sigma
      ## (i.e. a vector with 3 elements).
      estSigmas <- sigest(as.matrix(x), na.action = na.omit, scaled = TRUE)
      sigmas <- c(estSigmas[2], mean(estSigmas[-1]), mean(estSigmas[-2]), mean(estSigmas[-3]))
    } else {
      sigmas <- sigmaValues
    }

    if (is.null(cValues)) {
      C <- 2 ^((1:len) - 3)
    } else {
      C <- cValues
    }
    expand.grid(sigma = sigmas, C = C)
  }

  fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
    ksvm(x = as.matrix(x), y = y,
      kernel = laplacedot, # rbfdot,
      kpar = list(sigma = param$sigma),
      C = param$C,
      ...)
  }

  pred <- function(modelFit, newdata, preProc = NULL, submodels = NULL) {
    predict(modelFit, newdata)
  }

  prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL) {
    predict(modelFit, newdata, type='prob')
  }

  sortingFunc <- function(x) {
    x[order(x$C),]
  }

  # We are lucky that kernlab has this class level extraction function implemented for us.
  # party:::ctree would have to use: function(x) levels(x@data@get('response')[,1])
  levelsFunc <- function(x) lev(x)

  modelComponents <- list(
      type = 'Classification',
      library = 'kernlab',
      loop = NULL,
      parameters = parameters,
      grid = grid,
      fit = fit,
      predict = pred,
      prob = prob,
      sort = sortingFunc,
      levels = levelsFunc # only used for classification models using S4 methods
  )
}


## Cross-validates SVM with Laplacian kernel using caret and prints the best model parameters.
## Returns the prediction for the validation set.
cross_validate_svm_laplacian <- function(partition, formula) {
  modelComponents <- build_svm_laplacian_model_config()
  trainedModel <-
      train_with_caret(modelComponents, partition$trainingSet, formula, 'SVM with Laplacian')
  print(trainedModel, digits = 3)
  plot(trainedModel,  scales = list(x = list(log = 2)))
  predict_with_model(trainedModel, partition$validationSet, 'SVM with Laplacian')
}


## Predicts the unknowns of the test set using ksvm with Laplacian kernel and preset meta params.
predict_with_ksvm_laplacian <-
    function(training, test, formula, label, threshold = 0.5) {
  modelComponents <- build_svm_laplacian_model_config(sigmaValues=c(0.0273), cValues = c(8))
  train_and_predict_with_caret(modelComponents, training, test, formula, label, threshold)
}
