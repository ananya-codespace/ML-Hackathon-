# Per-model fact sheet (generated from results/comparison_results.json)

Numbers below are copied from saved result files. The heuristic flags are only prompts for you to look at the evidence - write your own interpretation for the report.


## Logistic Regression  (Linear / probabilistic)

- Hyperparameters (initial values): C=1.0, max_iter=500, imbalance=none
- Objective / loss: L2-regularised binary log loss (convex, solved by L-BFGS)
- Student-test: Top-1 = 0.8308, macro F1 = 0.8901, precision = 0.8138, recall = 0.8024, log loss = 0.1353
- Train (in-sample): Top-1 = 0.8251, macro F1 = 0.8861, log loss = 0.1441
- Grouped CV (5 folds by trip): macro F1 = 0.8843 +/- 0.0199, Top-1 = 0.8235 +/- 0.0417
- Timing: train 2.25s, inference 0.012s (0.1 us/row)
- Over/underfitting evidence: train->CV macro-F1 gap +0.0019; CV->student-test gap -0.0058
- Heuristic flags: none triggered

## Gaussian Naive Bayes  (Generative probabilistic)

- Hyperparameters (initial values): var_smoothing=1e-09, imbalance=none
- Objective / loss: Maximise class-conditional Gaussian likelihood (closed-form, no iterative loss)
- Student-test: Top-1 = 0.8135, macro F1 = 0.8280, precision = 0.6045, recall = 0.8572, log loss = 0.9173
- Train (in-sample): Top-1 = 0.8101, macro F1 = 0.8314, log loss = 0.8796
- Grouped CV (5 folds by trip): macro F1 = 0.8301 +/- 0.0171, Top-1 = 0.8081 +/- 0.0377
- Timing: train 1.65s, inference 0.017s (0.1 us/row)
- Over/underfitting evidence: train->CV macro-F1 gap +0.0013; CV->student-test gap +0.0021
- Heuristic flags: none triggered

## Random Forest  (Bagged tree ensemble)

- Hyperparameters (initial values): n_estimators=150, max_depth=None, min_samples_leaf=5, max_features=sqrt, max_samples=0.25, imbalance=none
- Objective / loss: Gini impurity reduction per split; probability = mean of tree leaf frequencies
- Student-test: Top-1 = 0.8412, macro F1 = 0.9054, precision = 0.8097, recall = 0.8636, log loss = 0.1050
- Train (in-sample): Top-1 = 0.8686, macro F1 = 0.9224, log loss = 0.0868
- Grouped CV (5 folds by trip): macro F1 = 0.9007 +/- 0.0228, Top-1 = 0.8329 +/- 0.0426
- Timing: train 46.87s, inference 0.360s (2.4 us/row)
- Over/underfitting evidence: train->CV macro-F1 gap +0.0217; CV->student-test gap -0.0047
- Heuristic flags: none triggered

## HistGradientBoosting  (Gradient boosting)

- Hyperparameters (initial values): learning_rate=0.1, max_iter=200, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0.0, imbalance=none
- Objective / loss: Binary log loss, minimised stage-wise with histogram-binned trees
- Student-test: Top-1 = 0.8385, macro F1 = 0.9030, precision = 0.8113, recall = 0.8524, log loss = 0.1029
- Train (in-sample): Top-1 = 0.8452, macro F1 = 0.9082, log loss = 0.1054
- Grouped CV (5 folds by trip): macro F1 = 0.8993 +/- 0.0218, Top-1 = 0.8308 +/- 0.0416
- Timing: train 6.35s, inference 0.124s (0.8 us/row)
- Over/underfitting evidence: train->CV macro-F1 gap +0.0090; CV->student-test gap -0.0037
- Heuristic flags: none triggered

## MLP (2 hidden layers)  (Neural network)

- Hyperparameters (initial values): hidden=64, hidden2=32, alpha=0.0001, learning_rate_init=0.001, max_iter=100, imbalance=none
- Objective / loss: Binary log loss + L2 penalty (alpha), Adam optimiser, early stopping on a 10% validation split
- Student-test: Top-1 = 0.8383, macro F1 = 0.9012, precision = 0.8104, recall = 0.8469, log loss = 0.1052
- Train (in-sample): Top-1 = 0.8462, macro F1 = 0.9067, log loss = 0.1059
- Grouped CV (5 folds by trip): macro F1 = 0.8975 +/- 0.0222, Top-1 = 0.8302 +/- 0.0425
- Timing: train 123.44s, inference 0.047s (0.3 us/row)
- Over/underfitting evidence: train->CV macro-F1 gap +0.0092; CV->student-test gap -0.0038
- Heuristic flags: none triggered
