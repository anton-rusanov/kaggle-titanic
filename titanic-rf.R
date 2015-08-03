library(Amelia)
library(caret)
library(e1071)
library(gbm)
library(ggplot2)
library(glmnet)
library(gplots)
library(kernlab)
library(party)
library(plyr)
library(pROC)
library(randomForest)
library(ROCR)
library(rpart)
library(stats)

# Set seed for reproducibility
set.seed(85431)

prepare_data <- function() {
  print('Started preparing data')
  train_column_types <- c(
    'integer',   # PassengerId
    'factor',    # Survived 
    'factor',    # Pclass
    'character', # Name
    'factor',    # Sex
    'numeric',   # Age
    'integer',   # SibSp
    'integer',   # Parch
    'character', # Ticket
    'numeric',   # Fare
    'character', # Cabin
    'factor'     # Embarked
  )
  
  test_column_types <- train_column_types[-2]     ## no Survived column in test.csv
  
  missing_types <- c('NA', '')
  
  if (!file.exists('train.csv') || !file.exists('test.csv')) {
    stop(paste('train.csv or test.csv not found in working directory ', getwd()))
  }
  
  train_data <- read.csv('train.csv', 
                         colClasses=train_column_types,
                         na.strings=missing_types)
  
  test_data <- read.csv('test.csv', 
                        colClasses=test_column_types,
                        na.strings=missing_types)
  
  # All data, both training and test set
  all_data <- rbind(train_data, cbind(test_data, Survived=0))

  ## Map missing data by feature
  #missmap(all_data, main='Titanic Training Data - Missings Map', legend=FALSE)

  all_data$SurvivedYn <- create_survivedYn(all_data)

  all_data$Title <- extract_titles(all_data)
  
  all_data$Title <- consolidate_titles(all_data)
  
  all_data <- filling_missing_entries(all_data)

  all_data$FamilyId <- create_family_id(all_data)

  all_data$FamilySize <- create_family_size(all_data)

  all_data$FamilySizeFactor <- create_family_size_factor(all_data)

  all_data$WithSameTicket <- count_people_with_same_ticket(all_data)

  all_data$RealFare <- calculate_real_fare(all_data)

  all_data$RealFareScaled <- scale_and_center_real_fare(all_data)

  all_data$AgeScaled <- scale_and_center_age(all_data)

  all_data$AgeFactor <- create_age_factor(all_data)

  all_data <- with_binary_columns(all_data)

  all_data$Mother <- create_is_mother(all_data)

  print('Finished preparing data')

  write.csv(all_data, file='all-data.csv', row.names=FALSE, quote=FALSE)

  return (all_data)
}


## Createss a Y/N factor column for survival. Those options can be used as data.frame columns, which is
## important for the SVM implementation we are using.
create_survivedYn <- function(data) {
  data$SurvivedYn <- ifelse(data$Survived == 1, 'Y', 'N')
  data$SurvivedYn <- as.factor(data$SurvivedYn)
  return (data$SurvivedYn)
}


## Extracts honorific (i.e. title) from the Name feature
extract_titles <- function(data) {
  print('Extracting titles')
  title.dot.start <- regexpr('\\,[A-Z ]{1,20}\\.', data$Name, TRUE)
  title.comma.end <- title.dot.start + attr(title.dot.start, 'match.length') - 1
  data$Title <- substr(data$Name, title.dot.start + 2, title.comma.end - 1)
  return (data$Title)
}


## Assigns a new title value to old title(s)
change_titles <- function(data, old.titles, new.title) {
  for (honorific in old.titles) {
    data$Title[which(data$Title == honorific)] <- new.title
  }
  return (data$Title)
}


## Title consolidation
consolidate_titles <- function(data) {
  print('Consolidating titles')
  data$Title <- change_titles(data,
      c('Capt', 'Col', 'Don', 'Dr', 'Jonkheer', 'Major', 'Rev', 'Sir'), 'Noble')
  data$Title <- change_titles(data, c('the Countess', 'Ms', 'Lady', 'Dona'), 'Mrs')
  data$Title <- change_titles(data, c('Mlle', 'Mme'), 'Miss')
  data$Title <- as.factor(data$Title)
  return (data$Title)
}


## Fills missing entries with median or predicted values.
filling_missing_entries <- function(all_data) {
  print('Filling missing entries')
  # Passenger on row 62 and 830 do not have a value for embarkment.
  # Since many passengers embarked at Southampton, we give them the value S.
  # We code all embarkment codes as factors.
  all_data$Embarked[c(62,830)] = 'S'
  all_data$Embarked <- factor(all_data$Embarked)

  # Passenger on row 1044 has an NA Fare value. Let's replace it with the median fare value.
  all_data$Fare[1044] <- median(all_data$Fare, na.rm=TRUE)

  # How to fill in missing Age values?
  # We make a prediction of a passengers Age using the other variables and a decision tree model.
  # This time you give method='anova' since you are predicting a continuous variable.
  predicted_age <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title,
                         data=all_data[!is.na(all_data$Age),], method='anova')
  predicted_age_values <- predict(predicted_age, all_data[is.na(all_data$Age),])
  all_data$Age[is.na(all_data$Age)] <- predicted_age_values
#  print('Predicted age:')
#  print(predicted_age_values)


  return (all_data)
}


## Creates family size feature
create_family_size <- function(data) {
  print('Adding FamilySize')
  return (data$SibSp + data$Parch + 1)
}


## Create FamilySizeFactor column
create_family_size_factor <- function(data) {
  print('Adding FamilySizeFactor')
  data$FamilySizeFactor <- ifelse(data$FamilySize <= 4,
      ifelse(data$FamilySize == 1, 'Single', 'Small'),
      'Large')

  data$FamilySizeFactor <- as.factor(data$FamilySizeFactor)
  return (data$FamilySizeFactor)
}


## Creates FamilyId factor feature.
## The feature is currently unused in our models, though found useful by other researchers.
create_family_id <- function(data) {
  data$Surname <- sapply(data$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})
  data$FamilyID <- paste0(as.character(data$FamilySize), data$Surname)
  data$FamilyID[data$FamilySize <= 2] <- 'Small'
  # Delete erroneous family IDs
  famIDs <- data.frame(table(data$FamilyID))
  famIDs <- famIDs[famIDs$Freq <= 2,]
  data$FamilyID[data$FamilyID %in% famIDs$Var1] <- 'Small'
  # Convert to a factor
  data$FamilyID <- factor(data$FamilyID)
  return (data$FamilyID)
}


## Counts people who have the same ticket number.
# TODO: Combine this with the family size?
# TODO: This could work the other way: AdultsOnTicket = TicketFare/Fare(Pclass)
count_people_with_same_ticket <- function(data) {
  print('Adding WithSameTicket')
  ave(data$PassengerId, data[, 'Ticket'], FUN=length)
}


## Calculates the ticket fare divided by the number of people on that ticket.
calculate_real_fare <- function(data) {
  print('Adding RealFare')
  return (data$Fare / data$WithSameTicket)
}


## Scales and centers RealFare using caret.preProcess.
scale_and_center_real_fare <- function(data) {
  return (scale_and_center_column(data, 'RealFare'))
}


## Scales and centers Age using caret.preProcess.
scale_and_center_age <- function(data) {
  return (scale_and_center_column(data, 'Age'))
}


## Scales and centers column 'columnName'.
scale_and_center_column <- function(data, columnName) {
  print(paste('Scaling and centering', columnName))
  columnValues <- as.matrix(data[, columnName])
  # TODO: Should scaling and centering be done only for the train set, and then applied to all?
  # Slides 30-32 of this doc authors recommend that:
  # http://www.r-project.org/nosvn/conferences/useR-2013/Tutorials/kuhn/user_caret_2up.pdf
  # But I don't get why this is better.
  preProcessValues <- preProcess(columnValues, method = c('center', 'scale'))
  return (predict(preProcessValues, columnValues))
}


## Creates a factor of age data
create_age_factor <- function(data) {
  print('Creating a factor for Age')
  data$AgeFactor <- ifelse(data$Age <= 65,
      ifelse(data$Age <= 16, 'Child', 'Adult'),
      'Elderly')
  data$AgeFactor <- as.factor(data$AgeFactor)
  return (data$AgeFactor)
}


## Creates binary columns for existing factor features, for logistic regression
with_binary_columns <- function(data) {
  print('Adding binary columns')
  # FamilySizeFactor
  data$FamilySizeFactorSingle <- as.numeric(data$FamilySizeFactor == 'Single')
  data$FamilySizeFactorSmall <- as.numeric(data$FamilySizeFactor == 'Small')
  data$FamilySizeFactorLarge <- as.numeric(data$FamilySizeFactor == 'Large')

  # Pclass
  data$Pclass1 <- as.numeric(data$Pclass == 1)
  data$Pclass2 <- as.numeric(data$Pclass == 2)
  data$Pclass3 <- as.numeric(data$Pclass == 3)

  # Sex
  data$SexFemale <- as.numeric(data$Sex == 'female')

  # Embarked
  data$EmbarkedC <- as.numeric(data$Embarked == 'C')
  data$EmbarkedQ <- as.numeric(data$Embarked == 'Q')
  data$EmbarkedS <- as.numeric(data$Embarked == 'S')

  # Title
  data$TitleMaster = as.numeric(data$Title == 'Master')
  data$TitleMiss = as.numeric(data$Title == 'Miss')
  data$TitleMr = as.numeric(data$Title == 'Mr')
  data$TitleMrs = as.numeric(data$Title == 'Mrs')
  data$TitleNoble = as.numeric(data$Title == 'Noble')

  # AgeFactor
  data$AgeFactorChild = as.numeric(data$AgeFactor == 'Child')
  data$AgeFactorAdult = as.numeric(data$AgeFactor == 'Adult')
  data$AgeFactorElderly = as.numeric(data$AgeFactor == 'Elderly')

  return (data)
}


## Writes the given solution to a file.
write_solution <- function(prediction, method, suffix, passengerIds) {
  solution <- data.frame(PassengerId = passengerIds, Survived = prediction)
  write.csv(solution, file=paste0('titanic-', method, '-', suffix, '.csv'), row.names=FALSE, quote=FALSE)
  return (solution)
}


## Creates a binary column indicating if the person is a mother.
## Requires that create_age_factor(..) and with_binary_columns(..) are called before.
create_is_mother <- function(data) {
  print('Creating binary column Mother')
  data$Mother <- (data$Parch > 0 & data$AgeFactor == 'Adult' & data$TitleMrs)
  return (data$Mother)
}


## TODO: Implement it, currently not functional.
## Trains a Logistic Regression model and predicts 'Survived' for the test set.
predict_with_logistic_regression <- function(train, test) {
  print('Starting LR model building')
  lr_columns = c('Pclass1', 'Pclass2', 'Pclass3',
      'SexFemale', 'AgeScaled', 'RealFareScaled',
      'FamilySizeFactorSingle', 'FamilySizeFactorSmall', 'FamilySizeFactorLarge',
      'EmbarkedC', 'EmbarkedQ', 'EmbarkedS',
      'TitleMaster', 'TitleMiss', 'TitleMr', 'TitleMrs', 'TitleNoble')
  x_train <- as.matrix(train[, lr_columns])
  y_train <- as.matrix(train[, 'Survived'])
  # TODO: Set up logistic regression prediction, when features are normalized. Use as.matrix?

#  lr = glmnet(x=x_train, y=y_train, alpha=1, family='binomial')
#  plot(lr)
#  print('Starting LR prediction')
#  lr_prediction = predict(lr, newx = test[, lr_columns], type = 'class')
#  lr_solution <- data.frame(PassengerId = test$PassengerId, Survived = lr_prediction)
#  write.csv(lr_solution, file='titanic-lr.csv', row.names=FALSE, quote=FALSE)
#  return (lr_solution)
}


## TODO: Try to use this
## Trains a Logistic Regression model and predicts 'Survived' for the test set, and
## runs 10-fold cross-validation on the model.
predict_with_predict_with_logistic_regression2 <- function(train, test) {
  folds <- createFolds(train$Survived, k=10)
  cv_result <- lapply(folds, function(x) {
    train <- train[-x,]
    test <- train[x,]
    train.glm <- glm(
        Survived ~ Pclass + Sex + Age + I(Age^2) + SibSp + I(SibSp^2) +
            Pclass + I(Pclass^3) + Title + I(FamilySize^2),
        family = binomial,
        data = train)
    train.glm$xlevels[['Title']] <- union(train.glm$xlevels[['y']], levels(train$Title))
    glm.pred <- predict.glm(train.glm, newdata = test, type = 'response')
    survival.glm = ifelse(glm.pred > 0.5, 1, 0)
    actual <- test$Survived
    kappa <- kappa2(data.frame(actual, survival.glm))$value
    return (kappa)
  })
  cv_result
}


## Trains the Random Forest model and predicts 'Survived' for the test set.
predict_with_random_forest <- function(train, test, formula, suffix, threshold = 0.5) {
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
  prediction <- as.data.frame(predictionMatrix)
  prediction01 <- ifelse(prediction$Y >= threshold, 1, 0)
  write_solution(prediction01, 'rf', suffix, test$PassengerId)
  return (prediction)
}


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
  prediction <- data.frame(Y = predictionY, N = predictionN)
  prediction01 <- ifelse(prediction$Y >= threshold, 1, 0)
  write_solution(prediction01, 'ci', suffix, test$PassengerId)
  return (prediction)
}


## Trains the Support Vector Machine model and predicts 'Survived' for the test set.
predict_with_caret_svm <- function(train, test, formula, suffix, threshold = 0.5) {
  predict_with_caret('svmRadial', train, test, formula, suffix, threshold)
}


## Trains a Stochastic Gradient Boosting model and predicts 'Survived' for the test set.
predict_with_caret_gbm2 <- function(train, test, formula, suffix, threshold = 0.5) {
  predict_with_caret('gbm', train, test, formula, suffix, threshold)
}


predict_with_caret <- function(method, train, test, formula, suffix, threshold = 0.5) {
  print(paste('Starting building model', method, '-', suffix))
  fitControl <- trainControl(method = 'repeatedcv',
      number = 10,
      repeats = 3, # TODO Set 10
      classProbs = TRUE,
      summaryFunction = twoClassSummary)

  fit <- train(formula, data = train,
    method = method,
    trControl = fitControl,
    preProc = c('center', 'scale'),
    tuneLength = 8, # TODO: set 16
    metric = 'ROC',
    verbose = FALSE)

# ERROR:  undefined columns selected
#  plot(fit,
#      metric = 'Kappa',  # Only 'Accuracy' or 'Kappa' for classification
#      plotType = 'line', # scatter, level, line
#      #digits:
#      output = 'layered' # data, ggplot, layered
#  )

  #TODO: need splitted data to check!
#  confusionMatrix(fit,
#      train,
#      positive = 'Y')
#      # prevalence = 0.25 # fraction of positives agains

  print(paste('Starting prediction for ', method, '-', suffix))
  prediction <- predict(fit, test, type = 'prob')
  prediction01 <- ifelse(prediction$Y >= threshold, 1, 0)
  write_solution(prediction01, method, suffix, test$PassengerId)
  return (prediction)
}


cross_validate <- function(partition, formula) {
  parameters <- data.frame(parameter = c('C', 'sigma'),
      class = rep('numeric', 2),
      label = c('Cost', 'Sigma'))

  grid <- function(x, y, len = NULL) {
    library(kernlab)
    ## This produces low, middle and high values for sigma
    ## (i.e. a vector with 3 elements).
    sigmas <- sigest(as.matrix(x), na.action = na.omit, scaled = TRUE)
    expand.grid(sigma = mean(sigmas[-2]),
        C = 2 ^((1:len) - 3)
    )
  }

  fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
    ksvm(x = as.matrix(x), y = y,
      kernel = rbfdot,
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
      type = 'Classification', # TODO: c('Classification', 'Regression')
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

  fitControl <- trainControl(method = 'repeatedcv',
       ## 10-fold CV...
       number = 10,
       ## repeated ten times
       repeats = 10, # TODO: 10!
       classProbs = TRUE,
       verbose = TRUE)

  set.seed(825)
  Laplacian <- train(formula, data = partition$trainingSet,
                     method = modelComponents,
                     preProc = c('center', 'scale'),
                     tuneLength = 8,
                     trControl = fitControl,
                     prob.model = TRUE)
  print(Laplacian, digits = 3)

  plot(Laplacian,  scales = list(x = list(log = 2)))

  show_model_performance(partition,
      function(trainIgnored, test, formulaIgnored, suffixIgnored, thresholdIgnored) {
        predict(Laplacian, test, type='prob')
      },
      formula,
      'SVM with Laplacian',
      0.5)

  return (Laplacian)
}


predict_survival =  function() {
  all_data <- prepare_data()

  # Split the data back into a train set and a test set
  train <- all_data[1:891,]
  test <- all_data[892:1309,]
  
  predict_with_cv_gbm2(train, test, formulaMotherRealFareSameTicket(), '1')

  predict_with_caret_svm(train, test, formulaMotherRealFareSameTicket(), '1', 0.5)

  predict_with_caret_svm(train, test, formulaScaledMotherRealFareSameTicket(), 'scaled', 0.5)

  predict_with_logistic_regression(train, test)

  rf_base_solution <- predict_with_random_forest(train, test,
      formulaFamilySizeFactor(),
      'base', 0.5)

  cif_base_solution <- predict_with_conditional_inference_forest(train, test,
      formulaFamilySizeFactor(),
      'base', 0.5)

  cif_mother_solution <- predict_with_conditional_inference_forest(train, test,
      formulaMotherRealFareSameTicket(),
      'mother', 0.5)

  print('Done!')
}


show_cross_table_graph <- function(actual, predicted, label, threshold = 0.5) {
  print(paste('Showing CrossTable graph for', label))

  v <- rep(NA, length(predicted))
  v <- ifelse(predicted$Y >= threshold & actual$Survived == 1, 'TP', v)
  v <- ifelse(predicted$Y >= threshold & actual$Survived == 0, 'FP', v)
  v <- ifelse(predicted$Y < threshold & actual$Survived == 1, 'FN', v)
  v <- ifelse(predicted$Y < threshold & actual$Survived == 0, 'TN', v)

  df <- data.frame(real=as.factor(actual$Survived), prediction=predicted$Y)
  df$pred_type <- v

  fp <- nrow(df[which(df$pred_type == 'FP'),])
  fn <- nrow(df[which(df$pred_type == 'FN'),])

  ggplot(data=df, aes(x=real, y=prediction)) +
    geom_violin(fill=rgb(1,1,1,alpha=0.6), color=NA) +
    geom_jitter(aes(color=pred_type), shape=1) +
    geom_hline(yintercept=threshold, color='red', alpha=0.6) +
    scale_color_discrete(name='type', guide=FALSE) +
    ggtitle(sprintf('%s,\n FP=%d, FN=%d', label, fp, fn)) +
    ylim(c(-0.01,1.01))
}


## Creates data partition of given labeled dataset into training and validation sets and returns it
## as a list with entry names 'trainingSet' and 'validationSet'.
partition_labeled_dataset <- function(labeledDataSet) {
  print('Starting partitioning dataset')
  inTrainingSet <- createDataPartition(labeledDataSet$SurvivedYn, p = 0.85, list = FALSE)
  print(paste('Training set size:', length(inTrainingSet)))

  trainingSet <- labeledDataSet[inTrainingSet, ]
  validationSet <- labeledDataSet[-inTrainingSet, ]

  list(trainingSet = trainingSet, validationSet = validationSet)
}


show_model_performance <-
    function(partition, predictorFunction, formula, suffix, threshold = 0.5) {
  predicted <- predictorFunction(
      partition$trainingSet, partition$validationSet, formula, suffix, threshold)

  print('Plotting performance graphs')
  # TODO Plot graphs more compactly, so that they can be compared.
  actual <- partition$validationSet$Survived
  pred <- prediction(predicted$Y, actual);

  # Recall-Precision curve
  recallPrecisionPerf <- performance(pred, 'prec', 'rec')
  plot(recallPrecisionPerf, main = suffix)

  # ROC curve
  rocPerf <- performance(pred, 'tpr', 'fpr')
  plot(rocPerf, main = suffix)

  # ROC area under the curve
  auc <- performance(pred, 'auc')@y.values
  print(paste('AUC:', auc))

  # F1 score
  f1Perf <- performance(pred, 'f')
  plot(f1Perf, main = suffix)

  # RMSE
  rmse <- performance(pred, 'rmse')@y.values
  print(paste('RMSE:', rmse))

  show_cross_table_graph(partition$validationSet, predicted, suffix, threshold)
}


show_all_cross_table_graphs <- function() {
  all_data <- prepare_data()
  training <- all_data[1:891,]

  partition <- partition_labeled_dataset(training)

  setClass('Model', slots = list(
      name='character',
      method = 'function',
      formula = 'formula'))

  models <- c(
      new('Model', name = 'rf-mother',
          method = predict_with_random_forest,
          formula = formulaMotherRealFareSameTicket())

      , new('Model', name = 'rf-familysize',
          method = predict_with_random_forest,
          formula = formulaFamilySizeFactor())

      , new('Model', name = 'rf-realfare',
          method = predict_with_random_forest,
          formula = formulaRealFareSameTicket())

      , new('Model', name = 'cif-mother',
          method = predict_with_conditional_inference_forest,
          formula = formulaMotherRealFareSameTicket())

#     , new('Model', name = 'cif-scaled-mother',
#          method = predict_with_conditional_inference_forest,
#          formula = formulaScaledMotherRealFareSameTicket())

#     , new('Model', name = 'svm-mother',
#          method = predict_with_caret_svm,
#          formula = formulaMotherRealFareSameTicket())
#
#     , new('Model', name = 'gbm-mother',
#          method = predict_with_caret_gbm2,
#          formula = formulaScaledMotherRealFareSameTicket())
#
#     , new('Model', name = 'svm-scaled-mother',
#          method = predict_with_caret_svm,
#          formula = formulaScaledMotherRealFareSameTicket())
  )

  graphs <- lapply(models, function(model) {
    show_model_performance(partition, model@method, model@formula, paste0(model@name, '-graph'))
  })

  pushViewport(viewport(layout = grid.layout(1, length(graphs))))

  for (i in 1:length(graphs)) {
    print(graphs[[i]], vp = viewport(layout.pos.row = 1, layout.pos.col = i))
  }

  print('Done!')
}


formulaScaledMotherRealFareSameTicket <- function() {
  SurvivedYn ~
      Pclass + Sex + AgeScaled + RealFareScaled +
      Embarked + Title + FamilySizeFactor + WithSameTicket + Mother
}


formulaMotherRealFareSameTicket <- function() {
  SurvivedYn ~
      Pclass + Sex + Age + RealFare +
      Embarked + Title + FamilySizeFactor + WithSameTicket + Mother
}


formulaFamilySizeFactor <- function() {
  SurvivedYn ~
      Pclass + Sex + Age + Fare + Embarked + Title + FamilySizeFactor
}


formulaRealFareSameTicket <- function() {
  SurvivedYn ~
      Pclass + Sex + Age + RealFare +
      Embarked + Title + FamilySizeFactor + WithSameTicket
}


#predict_survival()

# TODO
#- DONE Exploratory analysis in Excel
#- DONE Scale and center numeric features, esp. Fare
#- Set up cross-validation
#- Use ROC, accuracy, or Kappa as benchmark?
#- Use more family-related insight (how many of them survived/perished/unknown?) kNN? Special rules.
#- LR
#- Use cabin data
#- Ensemble the models using probabilistic approach
#- Embarked is another important feature that dictates the survival of female??
#- investigate caretEnsemble? Ada Boost?
#- log(fare + 0.23)?
#- Use PCA to reduce dimensions and get rid of multicollinearity
#- Use ridge regression algorithm?
#- results <- resamples(list(LVQ=modelLvq, GBM=modelGbm, SVM=modelSvm))

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