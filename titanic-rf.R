
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
    stop(paste("train.csv or test.csv not found in working directory ", getwd()))
  }
  
  train_data <- read.csv('train.csv', 
                         colClasses=train_column_types,
                         na.strings=missing_types)
  
  test_data <- read.csv('test.csv', 
                        colClasses=test_column_types,
                        na.strings=missing_types)
  
  # All data, both training and test set
  all_data <- rbind(train_data, cbind(test_data, Survived=0))
  
  print('Extracting titles')
  all_data$Title <- get_title(all_data)
  
  print('Consolidating titles')
  all_data$Title <- consolidate_titles(all_data)
  
  print('Munging data')
  all_data <- munge_data(all_data)
  
  print('Adding FamilySize')
  all_data$FamilySize <- all_data$SibSp + all_data$Parch + 1

  print('Adding FamilySizeFactor')
  all_data$FamilySizeFactor <- create_family_size_factor(all_data)

  print('Adding binary columns')
  all_data <- with_binary_columns(all_data)

  print('Finished preparing data')
  
  return (all_data)
}

munge_data <- function(all_data) {
  # Passenger on row 62 and 830 do not have a value for embarkment. 
  # Since many passengers embarked at Southampton, we give them the value S.
  # We code all embarkment codes as factors.
  all_data$Embarked[c(62,830)] = "S"
  all_data$Embarked <- factor(all_data$Embarked)
  
  # Passenger on row 1044 has an NA Fare value. Let's replace it with the median fare value.
  all_data$Fare[1044] <- median(all_data$Fare, na.rm=TRUE)
  
  # How to fill in missing Age values?
  # We make a prediction of a passengers Age using the other variables and a decision tree model. 
  # This time you give method="anova" since you are predicting a continuous variable.
  predicted_age <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title,
                         data=all_data[!is.na(all_data$Age),], method="anova")
  all_data$Age[is.na(all_data$Age)] <- predict(predicted_age, all_data[is.na(all_data$Age),])
  
  return (all_data)
}

## function for extracting honorific (i.e. title) from the Name feature
get_title <- function(data) {
  title.dot.start <- regexpr("\\,[A-Z ]{1,20}\\.", data$Name, TRUE)
  title.comma.end <- title.dot.start + attr(title.dot.start, "match.length") - 1
  data$Title <- substr(data$Name, title.dot.start + 2, title.comma.end - 1)
  return (data$Title)
}   

## function for assigning a new title value to old title(s) 
change_titles <- function(data, old.titles, new.title) {
  for (honorific in old.titles) {
    data$Title[which(data$Title == honorific)] <- new.title
  }
  return (data$Title)
}

## Title consolidation
consolidate_titles <- function(data) {
  data$Title <- change_titles(data, 
      c("Capt", "Col", "Don", "Dr", "Jonkheer", "Major", "Rev", "Sir"), "Noble")
  data$Title <- change_titles(data, c("the Countess", "Ms", "Lady", "Dona"), "Mrs")
  data$Title <- change_titles(data, c("Mlle", "Mme"), "Miss")
  data$Title <- as.factor(data$Title)
  return (data$Title)
}

## Create FamilySizeFactor column
create_family_size_factor <- function(data) {
  data$FamilySizeFactor <- ifelse(data$FamilySize <= 4,
      (if (data$FamilySize == 1) 'Single' else 'Small'),
      'Large')
  data$FamilySizeFactor[data$FamilySize <= 4] <-
      ifelse(data$FamilySize[data$FamilySize <= 4] == 1, 'Single', 'Small')


  data$FamilySizeFactor <- as.factor(data$FamilySizeFactor)
  return (data$FamilySizeFactor)
}

## Create binary columns for existing factor features, for logistic regression
with_binary_columns <- function(data) {
  # FamilySizeFactor
  data$FamilySizeSingle <- as.numeric(data$FamilySizeFactor == 'Single')
  data$FamilySizeSmall <- as.numeric(data$FamilySizeFactor == 'Small')
  data$FamilySizeLarge <- as.numeric(data$FamilySizeFactor == 'Large')

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

  return (data)
}

predict_survival =  function() {
  
  library(Amelia)
  library(glmnet)
  library(party)
  library(randomForest)
  library(rpart)
  library(stats)
  
  all_data = prepare_data()
  
  ## Map missing data by feature
  #missmap(all_data, main="Titanic Training Data - Missings Map", legend=FALSE)
  
  # Split the data back into a train set and a test set
  train <- all_data[1:891,]
  test <- all_data[892:1309,]
  
  # Train set and test set structure
  #str(train)
  #str(test)
  
  # Dataset summaries and plots
  print(summary(all_data))
  print('')
  print(summary(train))
  mosaicplot(train$FamilySize ~ train$Survived, xlab="FamilySize", ylab="Survived")
  
  # Set seed for reproducibility
  set.seed(111)
  
  print('Starting RF model building')
  # Apply the Random Forest Algorithm
  
  my_forest <- randomForest(
      as.factor(Survived) ~
          Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySizeFactor,
      train,
      importance = TRUE,
      ntree = 2000)

  print('Starting prediction')
  # Make your prediction using the test set
  rf_prediction <- predict(my_forest, test)
  
  # Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
  rf_solution <- data.frame(PassengerId = test$PassengerId, Survived = rf_prediction)
  
  print('Starting CI model building')
  cond_inf_forest <- cforest(
      as.factor(Survived) ~ Pclass + Sex + Age + Fare + Embarked + Title + FamilySizeFactor,
      data = train, controls=cforest_unbiased(ntree=2000, mtry=3))

  print('Starting CI prediction')
  cond_inf_prediction <- predict(cond_inf_forest, test, OOB=TRUE, type = "response")
  cond_inf_solution <- data.frame(PassengerId = test$PassengerId, Survived = cond_inf_prediction)
  
  # Write your solution away to a csv file with the name rf_solution.csv
  write.csv(rf_solution, file="titanic-rf.csv", row.names=FALSE, quote=FALSE)
  write.csv(cond_inf_solution, file="titanic-ci.csv", row.names=FALSE, quote=FALSE)
  
  print('Done!')
}

predict_survival()
