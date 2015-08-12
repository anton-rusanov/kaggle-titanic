# Set seed for reproducibility
set.seed(1963507)

library(e1071)
library(plyr)
library(stats)

source('prepare-data.r')
source('formulas.r')
source('commons.r')
source('logistic-regression.r')
source('random-forest.r')
source('conditional-inference-forest.r')
source('caret-methods.r')
source('ksvm.r')
source('performance-views.r')

setClass('Model', slots = list(
    name='character',
    method = 'function',
    formula = 'formula'))


## Predicts survival using all models returned by list_all_models().
predict_survival =  function() {
  all_data <- prepare_data()
  # Split the data back into a train set and a test set
  train <- all_data[1:891,]
  test <- all_data[892:1309,]

  models <- list_all_models()
  lapply(models, function(model) {
    prediction <- model@method(train, test, model@formula, model@name)
    write_solution(prediction, model@name, test$PassengerId, 0.5)
  })
  print('Done!')
}


## Shows the model performance graphs for all models returned by list_all_models().
show_all_cross_table_graphs <- function() {
  all_data <- prepare_data()
  training <- all_data[1:891,]
  partition <- partition_labeled_dataset(training)

  models <- list_all_models()
  graphs <- lapply(models, function(model) {
    show_model_performance(partition, model@method, model@formula, model@name)
  })

  pushViewport(viewport(layout = grid.layout(1, length(graphs))))
  for (i in 1:length(graphs)) {
    print(graphs[[i]], vp = viewport(layout.pos.row = 1, layout.pos.col = i))
  }
  print('Done!')
}

list_all_models <- function() {
  c(
  #      new('Model', name = 'rf-mother',
  #          method = predict_with_random_forest,
  #          formula = formulaMotherRealFareSameTicket())
  #

#        new('Model', name = 'ksvm-laplacian-realfare',
#            method = predict_with_ksvm_laplacian,
#            formula = formulaRealFareSameTicket())

  #      , new('Model', name = 'ksvm-laplacian-mother',
  #          method = predict_with_ksvm_laplacian,
  #          formula = formulaScaledMotherRealFareSameTicket())

        new('Model', name = 'randomForest-realfare',
            method = predict_with_random_forest,
            formula = formulaRealFareSameTicket())

        ,new('Model', name = 'rf-realfare',
            method = predict_with_rf,
            formula = formulaRealFareSameTicket())
#        ,new('Model', name = 'rf-realfare-scaled',
#            method = predict_with_random_forest,
#            formula = formulaScaledRealFareSameTicket())
  #      , new('Model', name = 'cif-mother',
  #          method = predict_with_conditional_inference_forest,
  #          formula = formulaMotherRealFareSameTicket())

#       , new('Model', name = 'cif-scaled-mother',
#            method = predict_with_conditional_inference_forest,
#            formula = formulaScaledMotherRealFareSameTicket())

#       , new('Model', name = 'cif-realfare-scaled',
#            method = predict_with_conditional_inference_forest,
#            formula = formulaScaledRealFareSameTicket())
       , new('Model', name = 'cf-realfare',
            method = predict_with_cforest,
            formula = formulaRealFareSameTicket())

  #     , new('Model', name = 'svm-mother',
  #          method = predict_with_caret_svm,
  #          formula = formulaMotherRealFareSameTicket())
  #
  #     , new('Model', name = 'gbm-mother',
  #          method = predict_with_caret_gbm,
  #          formula = formulaScaledMotherRealFareSameTicket())
  #
  #     , new('Model', name = 'svm-scaled-mother',
  #          method = predict_with_caret_svm,
  #          formula = formulaScaledMotherRealFareSameTicket())
    )
}

#predict_survival()

# TODO
#- DONE Exploratory analysis in Excel
#- DONE Scale and center numeric features, esp. Fare
#- DONE Set up cross-validation
#- DONE Use ROC, __accuracy__, or Kappa as benchmark?
#- Use PCA or Partial Least Squares (PLS) to reduce dimensions and get rid of m15y, see prcomp
#- Naive Bayes, since it tends to work better on small training sets
#- Logistic Regression
#- Ridge Regression
#- Use more family-related insight (how many of them survived/perished/unknown?) kNN? Special rules.
#- Use cabin data
#- Ensemble the models using probabilistic approach
#- Embarked is another important feature that dictates the survival of female??
#- investigate caretEnsemble? Ada Boost?
#- log(fare + 0.23)?
#- results <- resamples(list(LVQ=modelLvq, GBM=modelGbm, SVM=modelSvm))
#- Try to use global baseline approach
#- Try to use "Reusable holdout set" approach: http://www.sciencemag.org/content/349/6248/636

#- Findings of exploratory analysis:
#--- Pre-process and use cabin numbers: split series and fix bad ones, like F E46.
#--- In 6-digit ticket numbers, the first digit is class number, validate and drop
#--- All 7-digit tickets start with 31, they are for class 3, drop the prefix
#--- A.4 and A.5 tickets are class 3 only
#--- All C.A. tickets are class 2 or 3, check more prefixes.
#--- Check survival as a function of the ticket series
#--- Investigate name spelling: Heikkinen, Honkanen, Heininen (ticket #s and ages are similar)
#--- Compare efficiency of scaled family size vs boolean columns for factors
#--- Age prediction sucks in pClass=3, peak around 28 yo.
#    Try this: https://inclass.kaggle.com/c/deloitte-tackles-titanic/forums/t/9841/getting-high-scores-without-looking-at-actual-data-set/51120#post51120
#--- survival ~ family size: 2=50%, 3 = 60%, 4=75%, 5=20%.