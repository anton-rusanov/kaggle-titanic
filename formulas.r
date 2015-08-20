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
      Pclass + Sex + Age + RealFare + Embarked + Title + FamilySizeFactor + WithSameTicket
}

formulaScaledRealFareSameTicket <- function() {
  SurvivedYn ~
      Pclass + Sex + AgeScaled + RealFareScaled +
      Embarked + Title + FamilySizeFactor + WithSameTicket
}

formulaScaledRealFare <- function() {
  SurvivedYn ~
      Pclass + Sex + AgeScaled + RealFareScaled +
      Embarked + Title + FamilySizeFactor
}

formulaRealFareSameTicketPclassInv <- function() {
  SurvivedYn ~
      PclassInv + Sex + Age + RealFare +
      EmbarkedOrder + Title + FamilySizeFactor + WithSameTicket
}

formulaPca <- function() {
  ~ PclassInv + AgeScaled + RealFareScaled + FamilySize + WithSameTicket + EmbarkedOrder +
      as.numeric(SexFemale)
}

