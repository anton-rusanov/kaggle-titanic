library(party)

## Trains the Conditional Inference Forest model and predicts 'Survived' for the test set.
predict_with_conditional_inference_forest <-
    function(train, test, formula, suffix, threshold = 0.5) {
  print(paste('Starting building model CI -', suffix))
  # TODO CI with and w/o WithSameTicket differ in 4 rows, but result in the same score.
  # TODO Run cross-validation to figure the better model!
  cond_inf_forest <- cforest(
      formula,
      data = train,
      controls=cforest_unbiased(ntree=1000, mtry=3, trace=TRUE))

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


