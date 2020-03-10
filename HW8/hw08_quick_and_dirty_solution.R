library(AUC)
library(onehot)
library(xgboost)

X_train <- read.csv("hw08_training_data.csv", header = TRUE)
Y_train <- read.csv("hw08_training_label.csv", header = TRUE)
X_test <- read.csv("hw08_test_data.csv", header = TRUE)

encoder <- onehot(X_train, addNA = TRUE, max_levels = Inf)
X_train_d <- predict(encoder, data = X_train)
X_test_d <- predict(encoder, data = X_test)

set.seed(421)
test_predictions <- matrix(0, nrow = nrow(X_test_d), ncol = ncol(Y_train))
colnames(test_predictions) <- colnames(Y_train)
test_predictions[,1] <- X_test[, 1]
for (outcome in 1:6) {
  valid_customers <- which(is.na(Y_train[,outcome + 1]) == FALSE)
  boosting_model <- xgboost(data = X_train_d[valid_customers, -1], label = Y_train[valid_customers, outcome + 1], nrounds = 20, objective = "binary:logistic")
  training_scores <- predict(boosting_model, X_train_d[valid_customers, -1])
  # AUC score for training data
  print(auc(roc(predictions = training_scores, labels = as.factor(Y_train[valid_customers, outcome + 1]))))
  test_scores <- predict(boosting_model, X_test_d[, -1])
  test_predictions[, outcome + 1] <- test_scores
}
write.table(test_predictions, file = "hw08_test_predictions.csv", row.names = FALSE, sep = ",")
