library(glmnet)

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
