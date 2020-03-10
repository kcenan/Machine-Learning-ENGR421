
## QUESTION 1-2 : Read data set and split data into test and train data set

data_set <- read.csv("hw05_data_set.csv")
x_all <- data_set$eruptions
y_all <- data_set$waiting

x_train <- x_all[1:150]
y_train <- y_all[1:150]
x_test <- x_all[151:272]
y_test <- y_all[151:272]

# QUESTION 3 - Implementing a decision tree regression algorithm using the following pre-pruning rule: 
# If a node has 𝑃 or fewer data points, convert this node into a terminal node and do not split further,
# where 𝑃 is a user-defined parameter.

DecisionTree <- function(n) {
  # create lists
  splits <- c()
  means <- c()
  
  # start with all point in root node
  node_indices <- list(1:length(x_train))
  is_terminal <- c(FALSE)
  need_split <- c(TRUE)
  
  # alghoritm which will cover solution
  while (1) {
    split_nodes <- which(need_split)
    if (length(split_nodes) == 0) {
      break
    }
    for (split_node in split_nodes) {
      data_indices <- node_indices[[split_node]]
      need_split[split_node] <- FALSE
      node_mean <- mean(y_train[data_indices])
      if (length(x_train[data_indices]) <= n) {
        is_terminal[split_node] <- TRUE
        means[split_node] <- node_mean
      } else {
        is_terminal[split_node] <- FALSE
        special_values <- sort(unique(x_train[data_indices]))
        split_positions <- (special_values[-1] + special_values[-length(special_values)]) / 2
        split_scores <- rep(0, length(split_positions))
        for (s in 1:length(split_positions)) {
          left_indices <- data_indices[which(x_train[data_indices] <= split_positions[s])]
          right_indices <- data_indices[which(x_train[data_indices] > split_positions[s])]
          total_error <- 0
          if (length(left_indices) > 0) {
            mean <- mean(y_train[left_indices])
            total_error <- total_error + sum((y_train[left_indices] - mean) ^ 2)
          }
          if (length(right_indices) > 0) {
            mean <- mean(y_train[right_indices])
            total_error <- total_error + sum((y_train[right_indices] - mean) ^ 2)
          }
          split_scores[s] <- total_error / (length(left_indices) + length(right_indices))
        }
        if (length(special_values) == 1) {
          is_terminal[split_node] <- TRUE
          means[split_node] <- node_mean
          next 
        }
        best_split <- split_positions[which.min(split_scores)]
        splits[split_node] <- best_split
        
        ###creating right and left nodes with selected splits
        left_indices <- data_indices[which(x_train[data_indices] < best_split)]
        node_indices[[2 * split_node]] <- left_indices
        is_terminal[2 * split_node] <- FALSE
        need_split[2 * split_node] <- TRUE
        
        right_indices <- data_indices[which(x_train[data_indices] >= best_split)]
        node_indices[[2 * split_node + 1]] <- right_indices
        is_terminal[2 * split_node + 1] <- FALSE
        need_split[2 * split_node + 1] <- TRUE
      }
    }
  }
  result <- list("splits"= splits, "means"= means, "is_terminal"= is_terminal)
  return(result)
}

# QUESTION 4 - Learn a decision tree by setting the pre-pruning parameter 𝑃 to 25

result <- DecisionTree(25)
node_splits <- result$splits
node_means <- result$means
is_terminal <- result$is_terminal

# function will give prediction result for each point :
get_prediction <- function(dp, is_terminal, node_splits, node_means){
  index <- 1
  while (1) {
    if (is_terminal[index] == TRUE) {
      return(node_means[index])
    } else {
      if (dp <= node_splits[index]) {
        index <- index * 2
      } else {
        index <- index * 2 + 1
      }
    }
  }
}

#plot train data, test data and fit in the figure
plot(x_train, y_train, type = "p", col = "blue", ylab = "waiting time to next eruption (min)", xlab = "Eruption time (min)")
points(x_test, y_test, type = "p", col= "red")

seq <- 0.01
data_points <- seq(from = min(x_all), to = max(x_all), by = seq)
for (b in 1:length(data_points)) {
  x_left <- data_points[b]
  x_right <- data_points[b+1]
  lines(c(x_left, x_right), c(get_prediction(x_left, is_terminal, node_splits, node_means), get_prediction(x_left, is_terminal, node_splits, node_means)), lwd = 1, col = "black")
  if (b < length(data_points)) {
    lines(c(x_right, x_right), c(get_prediction(x_left, is_terminal, node_splits, node_means), get_prediction(x_right, is_terminal, node_splits, node_means)), lwd = 1, col = "black") 
  }
}

# QUESTION 5 - Getting RMSE when p is equal to 25
y_test_predicted <- sapply(X=1:length(x_test), FUN = function(i) get_prediction(x_test[i], is_terminal, node_splits, node_means))
RMSE <- sqrt(sum((y_test - y_test_predicted) ^ 2) / length(y_test))
sprintf("RMSE is %s when P is 25", RMSE)

# QUESTION 6 - P versus RMSE
prunings <- c(5,10,15,20,25,30,35,40,45,50)
RMSEs <- sapply(X=prunings, FUN = function(p) {
  result <- DecisionTree(p)
  node_splits <- result$splits
  node_means <- result$means
  is_terminal <- result$is_terminal
  y_test_predicted <- sapply(X=1:length(x_test), FUN = function(i) get_prediction(x_test[i], is_terminal, node_splits, node_means))
  RMSE <- sqrt(sum((y_test - y_test_predicted) ^ 2) / length(y_test))
})

plot(prunings, RMSEs,
     type = "o", lwd = 1, las = 1, pch = 1, lty = 2,
     xlab = "Pre-prunning size (P)", ylab = "RMSE")
