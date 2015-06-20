
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
  
  test_column_types <- train_column_types[-2]     # # no Survived column in test.csv
  
  missing_types <- c('NA', '')
  
  if (!file.exists('train.csv') || !file.exists('test.csv')) {
    stop(paste("train.csv or test.csv not found in working directory", getwd()))
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
  all_data <- munge_data(all_data);
  
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
                              c("Capt", "Col", "Don", "Dr", "Jonkheer", "Lady", "Major", "Rev", "Sir"),
                              "Noble")
  data$Title <- change_titles(data, c("the Countess", "Ms"), "Mrs")
  data$Title <- change_titles(data, c("Mlle", "Mme"), "Miss")
  data$Title <- as.factor(data$Title)
  return (data$Title)
}

predict_survival =  function() {
  
  library(Amelia)
  library(randomForest)
  library(rpart)
  library(stats)
  
  all_data = prepare_data()
  
  ## Map missing data by feature
  #missmap(all_data, main="Titanic Training Data - Missings Map", legend=FALSE)
  
  # Split the data back into a train set and a test set
  train <- all_data[1:891,]
  test <- all_data[892:1309,]
  
  # Train set and test set
  #str(train)
  #str(test)
  
  # Set seed for reproducibility
  set.seed(111)
  
  print('Starting RF model building')
  # Apply the Random Forest Algorithm
  
  my_forest <- randomForest(
    as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title, 
    train, 
    importance = TRUE, 
    ntree = 10000) #TODO: set to 1000 

  print('Starting prediction')
    # Make your prediction using the test set
  my_prediction <- predict(my_forest, test)
  
  # Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
  my_solution <- data.frame(PassengerId = test$PassengerId, Survived = my_prediction)
  
  # Write your solution away to a csv file with the name my_solution.csv
  write.csv(my_solution, file="my_solution.csv", row.names=FALSE, quote=FALSE)
  print('Done!')
}

predict_survival();
