setwd("D:/Ph.D/Programming/Py/NoiseAwareBoost/R_scripts")

set.seed(1)
library(rpart)
n_tasks <- 10
n_outliers <- 2
N_data_per_task <- 1025
N_data_per_task_train <- 300

D <- 5

source("common_functions.R")

# We create data, and fit the common function and theta and plot results

data <- create_tasks_data(N_data_per_task, n_tasks, n_outliers)
result <- split_data(data, N_data_per_task_train)
data_train <- result$data_train
data_test <- result$data_test

# Matrix with the number of iterations of each stage

configurations <- matrix(c(
	200, 0, 0,
	100, 0, 100,
	50, 0, 150,
	150, 0, 50,
	75, 0, 125
	), 5, 3, byrow = TRUE)

best_configuration <- find_best_configuration(data_train, configurations, round(0.8 * N_data_per_task_train))

print("Best configuration training Multi-task Boosting")

print(best_configuration)

n_iterations_first_stage <- best_configuration[ 1 ]
n_iterations_second_stage <- best_configuration[ 2 ]
n_iterations_third_stage <- best_configuration[ 3 ]


ensemble <- fit_ensemble(data_train, n_iterations_first_stage, n_iterations_second_stage, n_iterations_third_stage)

print("Finished training Multi-task Boosting")
print(sigmoid(ensemble$theta))

errors <- rep(0, length(data_test$X))
sds <- rep(0, length(data_test$X))

for (i in 1 : length(data_test$X)) {
	output <- predict_ensemble(ensemble, i, data_test$X[[ i ]])$prediction
	errors[ i ] <- mean((output - data_test$Y[[ i ]])^2)
	sds[ i ] <- sd((output - data_test$Y[[ i ]])^2) / sqrt(length(output))
}

print("Average error per task")
print(errors)
print("Average sd per task")
print(sds)
print("Mean error Non-outlier tasks")
print(mean(errors[ - length(errors)]))
print("sd error Non-outlier tasks")
print(sd(errors[ - length(errors)]) / sqrt(length(errors[  1 : (n_tasks - n_outliers) ])))
print("Mean error")
print(mean(errors))
print("sd error")
print(sd(errors) / sqrt(length(errors)))




