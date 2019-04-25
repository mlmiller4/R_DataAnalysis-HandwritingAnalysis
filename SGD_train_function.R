data <- train_data_3_5
labels <- train_labels_3_5
alpha <- 0.5

library(permute)
library(pracma)

# Initialize theta with random values
theta <- runif(nrow(data), min=0, max=1)
theta_hat <- numeric(length(theta))

Loss_theta <- 0         # value of Loss functions L(theta) and L^(theta)
Loss_theta_hat <- 0

# do a repeating loop until convergence is found
repeat{
  Loss_theta_hat <- Loss_theta
  
  theta <- shuffle(theta)
  theta_hat <- theta
  
  i=1
  j=1
  
  while(i < ncol(data)){        # cycles from 1 to 12,665 for train_0_1
    while(j < length(theta)){     # cycles from 1 to 784 (since each image is 28x28)
      
      currentImage <- data[,i]
      currentLabel <- labels[,i]
      
      numerator <- (-1) * currentLabel * currentImage[j]
      dotprod <- dot(theta, currentImage)
      denominator <- exp(currentLabel * dotprod) + 1
      gradient <- alpha * (numerator/denominator)
      
      theta_hat[j] <- theta[j] - gradient
      
      j = j + 1
      
    }
    
    # Copy values of theta_hat to theta
    j=1
    while(j < length(theta)){
      theta[j] <- theta_hat[j]
      j = j+1
    }
    
    i = i+1
  }
  
  # Calcuate loss (L_theta): the summation (1 ->n) of log(1 + exp(-y(i) * <theta, x(i)>))
  # -------------------------------------------------------------------------------------
  i = 1
  summation = 0
  while(i<ncol(data)){
    
    currentImage <- data[,i]
    currentLabel <- labels[,i]
    
    expTerm <- exp((-1) * currentLabel * dot(theta, currentImage))
    
    summation = summation + log(1 + expTerm)
    
    i = i+1
  }
  
  Loss_theta <- summation    
  
  # Break repeat loop if abs(L_theta_hat - L_theta) > epsilon
  epsilon = 0.1
  if(abs((Loss_theta_hat - Loss_theta)) < epsilon){
    return(theta)
    break
  }
  
} # repeat loop end

