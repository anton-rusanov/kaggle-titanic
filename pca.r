library(ggplot2)


## Calculates principal components with given formula on the data.
calc_pc <- function(formula, data) {
  print('Starting PCA')
  prcomp(formula, data)
}


## Predicts the principal components for the newdata.
predict_pc <- function(prcompValues, newdata, ...) {
  print('Predicting PC values')
  predict(prcompValues, newdata, ...)
}


## Calculates and displays the projections to principal components in the given data.
show_pc <- function(data, formula = formulaPca()) {
  print('Starting show PC')
  pca <- calc_pc(formula, data)

  print(pca)
  print(pca_plot(pca, 'PC1', 'PC2'))
  print(pca_plot(pca, 'PC3', 'PC4'))
  print(pca_plot(pca, 'PC5', 'PC6'))
  pca
}


## Plots the projections of each feature onto the principal components.
pca_plot <- function(pca, byX = 'PC1', byY = 'PC2') {
  loadings <- data.frame(pca$rotation, .names = row.names(pca$rotation))

  theta <- seq(0, 2*pi, length.out = 100)
  circle <- data.frame(x = cos(theta), y = sin(theta))

  columnNameWithOffsetX <- paste(byX, '+ 0.2')
  columnNameWithOffsetY <- paste(byY, '- 0.1')
  p <- ggplot(circle, aes(x, y)) +
      geom_path() +
      geom_text(data=loadings,
          mapping=aes_string(x = columnNameWithOffsetX, y = columnNameWithOffsetY,
              label = '.names', color = '.names')) +
      geom_jitter(data = loadings, mapping = aes_string(x = byX, y = byY, color = '.names'), shape = 2) +
      coord_fixed(ratio = 1) +
      labs(x = byX, y = byY)
}

