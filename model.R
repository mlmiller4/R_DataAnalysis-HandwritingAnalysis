# Matthew L. Miller
# mmiller319

# Training function taking data, labels (y-values) and alpha
train <- function(data, labels, alpha){

  #library(permute)
  library(pracma)
  data_copy <- as.matrix(data)   # Make sure data is in matrix format
  labels_copy <- as.matrix(labels)
  
  # Initialize theta with random values
  theta <- runif(nrow(data_copy), min=0, max=1) 
  theta_mat <- as.matrix(theta)             # version of theta cast to a matrix
  theta_hat <- c()          

  epochsElapsed <- 0
  
  currentLoss <- 0       # current and previous loss terms
  previousLoss <- 0 
  
  iter <- 0
  
  while(TRUE){
    iter <- iter+1
    
    # set previous loss to be the loss from the last epoch
    previousLoss <- currentLoss
    
    # Add labels to data matrix, shuffle and separate labels from data
    data_copy <- as.matrix(rbind(data, labels))
    #print(sprintf("Iteration = %i", iter))
    #print(sprintf("nrow in data_copy = %i", nrow(data_copy)))
    
    # Pick a column at random w/o replacement
    currentSample <- as.matrix(data_copy[, sample(1:ncol(data_copy), 1, replace = FALSE)])

    # Break current sample into image and labels
    currentLabel <- currentSample[length(currentSample)]
    currentImage <- as.vector(currentSample[-length(currentSample)])
    
    # Debugging --------------------------------------
    #print(sprintf("Iteration = %i", iter))
    #print(sprintf("Length of theta = %i", length(theta)))
    #print(sprintf("Length of currentImage = %i", length(currentImage)))
    
    # Calculate inner produce w/ bias of 1
    #innerproduct <- as.vector(theta) %*% as.vector(currentImage)
    innerproduct <- dot(theta, currentImage)
    innerproduct <- innerproduct + 1
    
    # Gradient - from Chuck Cottrill
    #grad_frac <- 1 / (1 + exp(-currentLabel * innerproduct))
    #gradient <- alpha * currentLabel * currentImage * (1 - 1/grad_frac)
    
    # Gradient - Taken from CS229 Stanford Paper
    grad_logistic <- -((currentLabel * currentImage) / (1 + exp(currentLabel * innerproduct)))
    gradient <- alpha * grad_logistic
    
    #gradient <- (alpha * (-currentLabel) * currentImage) / (1 + exp(currentLabel * innerproduct))  # Works!
    #gradient <- (alpha * currentLabel * currentImage * exp(-currentLabel * innerproduct)) / (1 + exp(-currentLabel * innerproduct))
    
    # Calculate theta hat
    theta_hat <- theta - gradient
    #theta <- theta_hat
    
    # Calcuate loss: the summation (1 ->n) of log(1 + exp(-y(i) * <theta, x(i)>))
    #log_terms <- log(1 + exp(-currentLabel * innerproduct))
    #currentLoss <- sum(log_terms)
    
    # Loss function corrected on Page 3 of "Introduction to Logistic Regression" reading
    currentLoss <- 1 / (1 + exp(-currentLabel * innerproduct))
    
    epochsElapsed <- epochsElapsed + 1
    
    epsilon = 0.01
    
    # If difference between this epoch's loss and the last epoch's loss is less than epsilon, break the loop and return theta
    if(abs(currentLoss - previousLoss) < epsilon){
      print(sprintf("Completed in %i iterations.", epochsElapsed))
      return(unname(theta))
    }
    
    theta <- theta_hat
  }
}

predict <- function(theta, data){

  # Calculate P(Y = y|X = X) = 1/(1 + exp(y<theta, x>))
  # if this is greater than 0.5, make label +1, if less than 0.5 make label -1  
  #--------------------------------------------------------------------------
  
  # Calculate inner product of theta and the data columns
  #innerProd <- theta %*% as.matrix(data)
  theta <- as.matrix(theta)
  data <- as.matrix(data)
  #innerProd <- dot(theta, data)
  
  # Calc innerProd using for loop?
  innerProd <- numeric(ncol(data))
  
  for(i in 1:ncol(data)){
    currentImage <- data[,i]
    currentInnerProd <- dot(theta, currentImage)
    innerProd[i] <- currentInnerProd
  }
  
  # Calculate probabilites, set to 1 if greater than 0.5, -1 if less than 0.5
  probabilities <- 1 / (1 + exp(innerProd))
  labels <- ifelse(probabilities > 0.5, 1, -1)
  
  return(labels) 
}

accuracy <- function(labels, labels_pred){
  
  truth <- vector(mode = 'logical', length = length(labels))
  
  truth <- ifelse(labels == labels_pred, TRUE, FALSE)
  
  return(sum(truth)/length(labels))
}

model <- function(train_data, train_labels, test_data, test_labels, alpha){
  
  theta <- train(train_data, train_labels, alpha)
  
  prediction_train_labels <- predict(theta, train_data)
  
  train_acc <- accuracy(prediction_train_labels, train_labels)
  
  prediction_test_labels <- predict(theta, test_data)
  
  test_acc <- accuracy(prediction_test_labels, test_labels)
  
  retList <- list(theta, train_acc, test_acc)
  
  return(retList)
}
  
# Executed code using functions above
# -------------------------------------------------------------------------------

# Data Preprocessing - load MNIST data and divide into 0/1 and 3/5 datasets with features
# and labels

# Read in mnist_train.csv and mnist_test.csv
training <- read.csv(file="mnist_train.csv", header=FALSE, sep=",")
test <- read.csv(file="mnist_test.csv", header = FALSE, sep = ",")

# Partition the training set for 0,1 and 3,5
train_data_0_1 <- training[ which(training[785,] == 0 | training[785,] == 1)]
train_data_3_5 <- training[ which(training[785,] == 3 | training[785,] == 5)]
test_data_0_1 <- test[ which(test[785,] == 0 | test[785,] == 1)]
test_data_3_5 <- test[ which(test[785,] == 3 | test[785,] == 5)]

# Separate class labels (row 785)
test_labels_0_1 <- test_data_0_1[785,]
test_labels_3_5 <- test_data_3_5[785,]
train_labels_0_1 <- train_data_0_1[785,]
train_labels_3_5 <- train_data_3_5[785,]

# Remove row 785 from image data
test_data_0_1 <- test_data_0_1[-c(785), ]
test_data_3_5 <- test_data_3_5[-c(785), ]
train_data_0_1 <- train_data_0_1[-c(785), ]
train_data_3_5 <- train_data_3_5[-c(785), ]

# Re-assign labels as +1 or -1
test_labels_0_1 <- ifelse(test_labels_0_1 == 0, -1, 1)
test_labels_3_5 <- ifelse(test_labels_3_5 == 3, -1, 1)
train_labels_0_1 <- ifelse(train_labels_0_1 == 0, -1, 1)
train_labels_3_5 <- ifelse(train_labels_3_5 == 3, -1, 1)

# 1. Implementation
# ---------------------------------------------------------------
# Call train and predict functions using the 0/1 and 3/5 datasets
thetas_0_1 <- train(train_data_0_1, train_labels_0_1, 0.1)
predctedLabels_0_1 <- predict(thetas_0_1, test_data_0_1)

thetas_3_5 <- train(train_data_3_5, train_labels_3_5, 0.1)
predictedLabels_3_5 <- predict(thetas_3_5, test_data_3_5)

# Create rotation function to correct for image() function's rotation
# Taken from: https://www.r-bloggers.com/creating-an-image-of-a-matrix-in-r-using-image/
rotate <- function(x) t(apply(x, 2, rev))

# Get examples of correct and incorrect images for 0/1 dataset
image0a <- train_data_0_1[,1]
image0b <- train_data_0_1[,10]
image1a <- train_data_0_1[,12665]
image1b <- train_data_0_1[,12000]

matrix0a <- matrix(image0a, nrow=28, ncol=28, byrow=FALSE)
matrix0b <- matrix(image0b, nrow=28, ncol=28, byrow=FALSE)
matrix1a <- matrix(image1a, nrow=28, ncol=28, byrow=FALSE)
matrix1b <- matrix(image1b, nrow=28, ncol=28, byrow=FALSE)

image(rotate(matrix0a), col=gray(0:255/255))
image(rotate(matrix0b), col=gray(0:255/255))
image(rotate(matrix1a), col=gray(0:255/255))
image(rotate(matrix1b), col=gray(0:255/255))

# Get examples of correct and incorrect images from 3/5 dataset
image3a <- train_data_3_5[,1]
image3b <- train_data_3_5[,10]
image5a <- train_data_3_5[,11552]
image5b <- train_data_3_5[,11000]

matrix3a <- matrix(image3a, nrow=28, ncol=28, byrow=FALSE)
matrix3b <- matrix(image3b, nrow=28, ncol=28, byrow=FALSE)
matrix5a <- matrix(image5a, nrow=28, ncol=28, byrow=FALSE)
matrix5b <- matrix(image5b, nrow=28, ncol=28, byrow=FALSE)

image(rotate(matrix3a), col=gray(0:255/255))
image(rotate(matrix3b), col=gray(0:255/255))
image(rotate(matrix5a), col=gray(0:255/255))
image(rotate(matrix5b), col=gray(0:255/255))
#---------------------------------------------------------------------------


# 2. Modeling
# --------------------------------------------------------------------------

# alpha values
alphas <- c(0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.2)

# plot accuracies
accuracies_0_1 <- rep_len(0.463357, length.out = 10)
accuracies_3_5 <- rep_len(0.53102, length.out = 10)
plot(alphas, accuracies_0_1, type='l',col='red', ylab="Accuracy", xlab="alpha values", ylim=c(0,1))
lines(alphas, accuracies_3_5, type='l', col='blue')
title(main="Accuracies for 0_1 and 3_5")
legend('topright', inset=.05, legend=c("0 & 1", "3 & 5"), col=c('red', 'blue'), lty=1, cex=0.8)

# Get accuracies using alpha values and model() function
accuracies_0_1 <- c()
accuracies_3_5 <- c()

for(a in alphas){

  result1 <- model(train_data_0_1, train_labels_0_1, test_data_0_1, test_labels_0_1, a)
  accuracies_0_1 <- c(accuracies_0_1, result1[[3]])

  result2 <- model(train_data_3_5, train_labels_3_5, test_data_3_5, test_labels_3_5, a)
  accuracies_3_5 <- c(accuracies_3_5, result2[[3]])
}
