############################################################# 
## Description: This script implements the lasso
#############################################################

#####################################
##     Lasso solution path         ##
#####################################

myLasso <- function(X, Y, lambda_all){
  
  # Find the lasso solution path for various values of 
  # the regularization parameter lambda.
  # 
  # X: Matrix of explanatory variables.
  # Y: Response vector
  # lambda_all: Vector of regularization parameters. Make sure 
  # to sort lambda_all in decreasing order for efficiency.
  #
  # Returns a matrix containing the lasso solution vector 
  # beta for each regularization parameter.
  
  nIter    <- 100
  p        <- dim(X)[2]
  L        <- length(lambda_all)
  
  beta     <- matrix(rep(0, p), nrow = p)
  beta_all <- matrix(rep(0, p * L), nrow = p)
  R  <- Y
  ss <- rep(0, p)
  for(j in 1:p){
    ss[j] <- sum(X[,j]^2)
  }
  for(l in 1:L){
    lambda <- lambda_all[l]
    for(t in 1:nIter){
      for(j in 1:p){
        db <- sum(R * X[,j]) / ss[j]
        b  <- beta[j] + db
        b  <- sign(b) * max(0, abs(b) - lambda / ss[j])
        db <- b - beta[j]
        R  <- R - X[,j] * db
        beta[j] <- b
      }
    }
    beta_all[,l] <- beta
  }
  
  
  ## Function should output the matrix beta_all, the 
  ## solution to the lasso regression problem for all
  ## the regularization parameters. 
  ## beta_all is (p+1) x length(lambda_all)
  return(beta_all)
  
}


n <- 50
p <- 10
s <- 5
lambda_all <- seq(1000, 10, length.out = 10)
X <- matrix(rnorm(n * p), nrow = n, ncol = p)
Y <- X %*% matrix(c((1:s), rep(0, p-s)), nrow = p) + rnorm(n)
beta_all <- myLasso(X, Y, lambda_all)
par(mfrow=c(1,1))
matplot(t((matrix(rep(1,p), nrow = 1)) %*% abs(beta_all)),
        t(beta_all), type = "l")


