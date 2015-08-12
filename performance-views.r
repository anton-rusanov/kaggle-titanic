library(ggplot2)
library(gplots)
library(pROC)
library(ROCR)


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


## Shows the confusion graph for the given data.
show_cross_table_graph <- function(actual, predicted, label, threshold = 0.5) {
  print(paste('Showing CrossTable graph for', label))

  v <- rep(NA, length(predicted))
  v <- ifelse(predicted$Y >= threshold & actual$Survived == 1, 'TP', v)
  v <- ifelse(predicted$Y >= threshold & actual$Survived == 0, 'FP', v)
  v <- ifelse(predicted$Y < threshold & actual$Survived == 1, 'FN', v)
  v <- ifelse(predicted$Y < threshold & actual$Survived == 0, 'TN', v)

  df <- data.frame(real=as.factor(actual$Survived), prediction=predicted$Y)
  df$pred_type <- v
# TODO: Use this instead? confusionMatrix(fit, train, positive = 'Y', prevalence = 0.25)

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


## Shows the model performance graphs for predictorFunction.
show_model_performance <-
    function(partition, predictorFunction, formula, label, threshold = 0.5) {
  predicted <- predictorFunction(
      partition$trainingSet, partition$validationSet, formula, label)

  print('Plotting performance graphs')
  # TODO Plot graphs more compactly, so that they can be compared.
  actual <- partition$validationSet$Survived
  pred <- prediction(predicted$Y, actual);

  # Recall-Precision curve
  recallPrecisionPerf <- performance(pred, 'prec', 'rec')
  plot(recallPrecisionPerf, main = label)

  # ROC curve
  rocPerf <- performance(pred, 'tpr', 'fpr')
  plot(rocPerf, main = label)

  # ROC area under the curve
  auc <- performance(pred, 'auc')@y.values
  print(sprintf('AUC: %.3f', auc))

  # F1 score
  f1Perf <- performance(pred, 'f')
  plot(f1Perf, main = label)

  # RMSE
  rmse <- performance(pred, 'rmse')@y.values
  print(sprintf('RMSE: %.3f', rmse))

  show_cross_table_graph(partition$validationSet, predicted, label, threshold)
}


