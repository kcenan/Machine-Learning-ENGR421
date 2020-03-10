library(tree)

X_train <- read.csv("training_data.csv", header = TRUE)
X_test <- read.csv("test_data.csv", header = TRUE)

tree_model <- tree(TRX_COUNT ~ REGION + DAY + MONTH + YEAR + TRX_TYPE, data = X_train)
training_scores <- predict(tree_model, X_train)

print(tree_model)

plot(tree_model)
text(tree_model)

# mean absolute error for training data
mean(abs(training_scores - X_train$TRX_COUNT))

# root mean squared error for training data
sqrt(mean((training_scores - X_train$TRX_COUNT)^2))

test_scores <- predict(tree_model, X_test)
write.table(test_scores, file = "test_predictions.csv", row.names = FALSE, col.names = FALSE)
