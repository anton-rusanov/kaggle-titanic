My Heart Will Go On
===
Super-short instruction
---
1. Download train.csv and test.csv from [Kaggle website](https://www.kaggle.com/c/titanic/).
2. `source('titanic-main.r')`. It defines the main runners `predict_survival()` and
`show_all_cross_table_graphs()` and the models - combinations of algorithm methods and the formulas
 defining the list of variables used for training. That R file sources all others.
3. `show_all_cross_table_graphs()` will show the cross-graphs for all models.

   `predict_survival()` will write the actual predictions using each model, in submittable format.