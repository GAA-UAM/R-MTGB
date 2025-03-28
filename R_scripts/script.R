set.seed(1)
library(rpart)
N_data_per_task <- 250

D <- 2

sigmoid <- function(x) 1 / (1 + exp(-x))

plot_2d_function <- function(f) {
	x1 <- seq(-1, 1, length = 50)
	x2 <- seq(-1, 1, length = 50)
	z <- matrix(0, length(x1), length(x2))

	for (i in 1 : length(x1))
		for (j in 1 : length(x1)) 
			z[ i, j ] <- f(matrix(c(x1[ i ], x2[ j ]), 1, 2))

	contour(x1, x2, z)
}

create_random_function <- function() {

	length_scale <- 0.25 * D 

	N_random_Features <- 500

	w <- matrix(rnorm(N_random_Features * D), N_random_Features, D)
	b <- runif(N_random_Features) * 2 * pi

	theta <- rnorm(N_random_Features)

	f <- function(x) {
		apply(as.matrix(x), 1, function(x) {
			x <- x / length_scale
			phi <- sqrt(2) * cos(w %*% x + b) * 1 / sqrt(N_random_Features)
			sum(phi * theta)
		})
	}

	return(f)
}

create_tasks_data <- function(n_tasks = 8, n_outliers = 1) {

	X <- list()
	Y <- list()

	ch <- create_random_function()

	# First we create non-outlier tasks

	rh_list <- list()
	
	for (i in 1 : (n_tasks - n_outliers)) {

		rh <- create_random_function()
		rh_list[[ i ]] <- rh

		# Training data in [-1,1]

		x <- matrix(runif(N_data_per_task * D) * 2 - 1, N_data_per_task, D)
#		y <- ch(x)#0.9 * ch(x) + 0.1 * rh(x)
		y <- 0.9 * ch(x) + 0.1 * rh(x)

		X[[ i ]] <- x
		Y[[ i ]] <- y
	}

	# Next, we create outlier tasks

	for (i in (((n_tasks - n_outliers) + 1) : n_tasks)) {

		rh <- create_random_function()

		# Training data in [-1,1]

		x <- matrix(runif(N_data_per_task * D) * 2 - 1, N_data_per_task, D)
#		rh <- function(x) {-ch(x)}#rh(x)
		y <- rh(x)

		X[[ i ]] <- x
		Y[[ i ]] <- y

		rh_list[[ i ]] <- rh

	}

	return(list(X = X, Y = Y, ch = ch, rh = rh_list))
}


create_ensemble <- function(n_tasks) {

	# Creates empty ensemble with 0 as the initial guess

	learning_rate <- 1.0
	ensemble_members_mt <- list()
	ensemble_members_ch <- list()
	ensemble_members_rh <- list()
#	theta <- rep(0, n_tasks)
	theta <- rnorm(n_tasks) * 0.1

	for (i in 1 : n_tasks)
		ensemble_members_rh[[ i ]] <- list()
	
	return(list(ensemble_members_mt = ensemble_members_mt, ensemble_members_ch = ensemble_members_ch, theta = theta, 
		    learning_rate = learning_rate, ensemble_members_rh = ensemble_members_rh))
}

predict_ensemble <- function(ensemble, n_task, x) {

	# Computes predictions of the ensemble for a task 

	pred <- rep(0, nrow(x))

	if (length(ensemble$ensemble_members_mt) > 0) {
		for (i in 1 : length(ensemble$ensemble_members_mt))
			pred <- pred + predict(ensemble$ensemble_members_mt[[ i ]], data.frame(x = x)) * ensemble$learning_rate
	}

	ensemble_output_mt <- pred

	pred <- pred * 0.0

	if (length(ensemble$ensemble_members_ch_non_out) > 0) {
		for (i in 1 : length(ensemble$ensemble_members_ch_non_out))
			pred <- pred + predict(ensemble$ensemble_members_ch_non_out[[ i ]], data.frame(x = x)) * ensemble$learning_rate
	}

	ensemble_output_ch_non_out <- pred * (1 - sigmoid(ensemble$theta[ n_task ]))

	pred <- pred * 0.0

	if (length(ensemble$ensemble_members_ch_out) > 0) {
		for (i in 1 : length(ensemble$ensemble_members_ch_out))
			pred <- pred + predict(ensemble$ensemble_members_ch_out[[ i ]], data.frame(x = x)) * ensemble$learning_rate
	}

	ensemble_output_ch_out <- pred * sigmoid(ensemble$theta[ n_task ])

	# Now the task specific functions

	ensemble_output_rh <- pred * 0.0

	if (length(ensemble$ensemble_members_rh[[ n_task ]]) > 0) {
		for (i in 1 : length(ensemble$ensemble_members_rh[[ n_task ]]))
			ensemble_output_rh <- ensemble_output_rh + predict(ensemble$ensemble_members_rh[[ n_task ]][[ i ]], data.frame(x = x)) * ensemble$learning_rate
	}

	return(list(ensemble_output_mt = ensemble_output_mt, ensemble_output_ch_non_out = ensemble_output_ch_non_out,
		ensemble_output_ch_out = ensemble_output_ch_out,  ensemble_output_rh = ensemble_output_rh, 
		prediction = ensemble_output_mt + ensemble_output_ch_non_out + ensemble_output_ch_out + ensemble_output_rh))
}

update_ensemble_first_stage <- function(ensemble, data) {

	# Incorporates an ensemble element into ch
	# and updates theta

	loss <- 0
	loss_outlier <- 0
	losses <- rep(0, length(data$X))

	for (i in 1 : length(data$X)) {

		output <- predict_ensemble(ensemble, i, data$X[[ i ]])

		if (i == 1) {
			gradient_ch <- - (data$Y[[ i ]] - output$prediction)
		} else {
			gradient_ch <- c(gradient_ch, - (data$Y[[ i ]] - output$prediction))
		}

		loss <- loss + sum((output$prediction - data$Y[[ i ]])^2)

		if (i == 8)
			loss_outlier <- sum((output$prediction - data$Y[[ i ]])^2)

		losses[ i ] <- sum((output$prediction - data$Y[[ i ]])^2)
	}

	loss <- loss / (N_data_per_task * length(data$X))
	loss_outlier <- loss_outlier / N_data_per_task
	losses <- losses / N_data_per_task

	print(loss)
	print(losses)
	print(loss_outlier)

	# We fit predictor to the negative gradient of ch

	x <- data$X[[ 1 ]]
	for (i in 2 : length(data$X)) {
		x <- rbind(x, data$X[[ i ]])
	}

	data_to_train_predictor <- data.frame(x = x, y = -gradient_ch)
	model <- rpart(y~x.1+x.2, data = data_to_train_predictor, control = list(maxdepth = 1, minsplit = 1, cp = 0.0, xval = 0))
	ensemble$ensemble_members_mt[[ length(ensemble$ensemble_members_mt) + 1 ]] <- model

	return(ensemble)
}


update_ensemble_second_stage <- function(ensemble, data) {

	# Incorporates an ensemble element into ch
	# and updates theta

	gradient_theta <- rep(0, length(data$X))

	loss <- 0
	loss_outlier <- 0
	losses <- rep(0, length(data$X))

	for (i in 1 : length(data$X)) {

		output <- predict_ensemble(ensemble, i, data$X[[ i ]])

		if (i == 1) {
			gradient_ch_non_out <- - (data$Y[[ i ]] - output$prediction) * (1 - sigmoid(ensemble$theta[ i ]))
			gradient_ch_out <- - (data$Y[[ i ]] - output$prediction) * sigmoid(ensemble$theta[ i ])
		} else {
			gradient_ch_non_out <- c(gradient_ch_non_out, - (data$Y[[ i ]] - output$prediction) * (1 - sigmoid(ensemble$theta[ i ])))
			gradient_ch_out <- c(gradient_ch_out, - (data$Y[[ i ]] - output$prediction) * sigmoid(ensemble$theta[ i ]))
		}

		gradient_theta[ i ] <- sum((data$Y[[ i ]] - output$prediction) * output$ensemble_output_ch_non_out * sigmoid(ensemble$theta[ i ])) - 
			sum((data$Y[[ i ]] - output$prediction) * output$ensemble_output_ch_out * (1 - sigmoid(ensemble$theta[ i ])))

		loss <- loss + sum((output$prediction - data$Y[[ i ]])^2)

		if (i == 8)
			loss_outlier <- sum((output$prediction - data$Y[[ i ]])^2)

		losses[ i ] <- sum((output$prediction - data$Y[[ i ]])^2)
	}


	loss <- loss / (N_data_per_task * length(data$X))
	loss_outlier <- loss_outlier / N_data_per_task
	losses <- losses / N_data_per_task

	# We update theta

	ensemble$theta <- ensemble$theta - gradient_theta * ensemble$learning_rate

	print(loss)
	print(losses)
	print(loss_outlier)
	print(gradient_theta)

	# We fit predictor to the negative gradient of ch

	x <- data$X[[ 1 ]]
	for (i in 2 : length(data$X)) {
		x <- rbind(x, data$X[[ i ]])
	}

	data_to_train_predictor <- data.frame(x = x, y = -gradient_ch_non_out)
	model <- rpart(y~x.1+x.2, data = data_to_train_predictor, control = list(maxdepth = 1, minsplit = 1, cp = 0.0, xval = 0))
	ensemble$ensemble_members_ch_non_out[[ length(ensemble$ensemble_members_ch_non_out) + 1 ]] <- model


	data_to_train_predictor <- data.frame(x = x, y = -gradient_ch_out)
	model <- rpart(y~x.1+x.2, data = data_to_train_predictor, control = list(maxdepth = 1, minsplit = 1, cp = 0.0, xval = 0))
	ensemble$ensemble_members_ch_out[[ length(ensemble$ensemble_members_ch_out) + 1 ]] <- model

	return(ensemble)
}

update_ensemble_third_stage <- function(ensemble, data) {

	# Incorporates an ensemble element into rh for each task

	for (i in 1 : length(data$X)) {


		output <- predict_ensemble(ensemble, i, data$X[[ i ]])
	        gradient_rh <- - (data$Y[[ i ]] - output$prediction) 

		# We fit the predictor

		data_to_train_predictor <- data.frame(x = data$X[[ i ]], y = -gradient_rh)
		model <- rpart(y~x.1+x.2, data = data_to_train_predictor, control = list(maxdepth = 1, minsplit = 1, cp = 0.0, xval = 0))
		ensemble$ensemble_members_rh[[ i ]][[ length(ensemble$ensemble_members_rh[[ i ]]) + 1 ]] <- model
	}

	return(ensemble)
}

# We create data, and fit the common function and theta and plot results

data <- create_tasks_data()
ensemble <- create_ensemble(length(data$X))
n_iterations_first_stage  <- 3
n_iterations_second_stage  <- 3
n_iterations_third_stage  <- 3

# We carry out the first stage, where we update ch and theta

for (i in 1 : n_iterations_first_stage) {
	print(i)
	ensemble <- update_ensemble_first_stage(ensemble, data)
}


# We carry out the second stage, where we update ch and theta

for (i in 1 : n_iterations_first_stage) {
	print(i)
	ensemble <- update_ensemble_second_stage(ensemble, data)
	print(sigmoid(ensemble$theta))
}

## We plot the results
#
#curve(0.9 * data$ch(x), -1, 1, col = "black", lwd = 2, xlab = "x", ylab = "y", main = "After First Stage")
#colors = c("red", "green", "yellow", "blue", "pink", "orange", "brown", "purple")
#
#for (i in 1 : length(data$X)) {
#	x_test <- seq(-1, 1, length = 100)
#	y_test <- predict_ensemble(ensemble, i, x_test)
#	lines(x_test, y_test, col = colors[ i ], lwd = 2)
#	points(data$X[[ i ]], data$Y[[ i ]], col = colors[ i ], lwd = 2)
#}
#
#legend("topright", c("Common Function", 
#		     "task 1",
#		     "task 2",
#		     "task 3",
#		     "task 4",
#		     "task 5",
#		     "task 6",
#		     "task 7",
#		     "task 8"), col = c("black", "red", "green", "yellow", "blue", "pink", "orange", "brown", "purple"), lty = 1)
#
#
# The fit is robust since it explains well non-outlier tasks (the common function). If the update for theta 
# is frozen, the fit is much worse, showing the impact of the outlier task.

# We carry out the second stage, where we update rh 

for (i in 1 : n_iterations_third_stage) {
	print(i)
	ensemble <- update_ensemble_third_stage(ensemble, data)
}


## We plot the results in a new window
#
#x11()
#
#curve(0.9 * data$ch(x), -1, 1, col = "black", lwd = 2, xlab = "x", ylab = "y", main = "After Second Stage")
#colors = c("red", "green", "yellow", "blue", "pink", "orange", "brown", "purple")
#
#for (i in 1 : length(data$X)) {
#	x_test <- seq(-1, 1, length = 100)
#	y_test <- predict_ensemble(ensemble, i, x_test)
#	lines(x_test, y_test, col = colors[ i ], lwd = 2)
#	points(data$X[[ i ]], data$Y[[ i ]], col = colors[ i ], lwd = 2)
#}
#
#legend("topright", c("Common Function", 
#		     "task 1",
#		     "task 2",
#		     "task 3",
#		     "task 4",
#		     "task 5",
#		     "task 6",
#		     "task 7",
#		     "task 8"), col = c("black", "red", "green", "yellow", "blue", "pink", "orange", "brown", "purple"), lty = 1)


#for (i in 1 : length(data$X)) {
#	write.table(data$X[[ i ]], paste("x_train_task_", i, ".txt", sep = ""), row.names = F, col.names = F)
#	write.table(data$Y[[ i ]], paste("y_train_task_", i, ".txt", sep = ""), row.names = F, col.names = F)
#}



output <- predict_ensemble(ensemble, 1, data$X[[ 1 ]])
print(output$prediction[0:5])
print(ensemble$theta)
