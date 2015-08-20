library(Amelia)
library(caret)
library(rpart)

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

  all_data$PclassInv <- create_pclass_inverted(all_data)

  all_data$EmbarkedOrder <- create_embarked_order_numeric(all_data)

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
  print('Adding FamilyID')
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


## Creates a binary column indicating if the person is a mother.
## Requires that create_age_factor(..) and with_binary_columns(..) are called before.
create_is_mother <- function(data) {
  print('Creating binary column Mother')
  data$Mother <- (data$Parch > 0 & data$AgeFactor == 'Adult' & data$TitleMrs)
  return (data$Mother)
}


## Creates a numeric column with inverted Pclass.
create_pclass_inverted <- function(data) {
  print('Creating a inverted value for Pclass')
  PclassNum <- as.numeric(as.character(data$Pclass))
  #  x == 1 => 2 - 1 = 1
  #  x == 2 => 1.58 - 1 = 0.58
  #  x == 3 => 1 - 1 = 0
  (log2(5 - PclassNum) - 1)
}


## Creates a numeric feature from Embarked, corresponding to the number of the stop.
create_embarked_order_numeric <- function(data) {
  print('Creating a numeric value for Embarked, in order')
  ifelse(data$Embarked == 'S', 1., ifelse(data$Embarked == 'C', 2., 3.))
}
