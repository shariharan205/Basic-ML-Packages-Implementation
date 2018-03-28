############################################################# 
## This script implements linear regression 
## using Gauss-Jordan elimination in both plain and
## vectorized forms
#############################################################


###############################################
## Plain version of Gauss Jordan             ##
###############################################


myGaussJordan <- function(A, m){
  
  # Perform Gauss Jordan elimination on A.
  # 
  # A: a square matrix.
  # m: the pivot element is A[m, m].
  # Returns a matrix with the identity matrix 
  # on the left and the inverse of A on the right. 

  
  n <- dim(A)[1]
  B <- cbind(A, diag(rep(1,n)))

  
  for(k in 1:m)
  {
    a <- B[k,k]
    
    for(j in 1:(n*2))
    {
      B[k,j] <- B[k,j]/a
    }
    
    for(i in 1:n)
    {
      if(i != k)
      {
        b <- B[i,k]
        
        for(j in 1:(n*2))
          
          B[i,j] <- B[i,j] - b*B[k,j]
        
      }
    }
  }
  
  ## Function returns the matrix B

  return(B)
  
}

####################################################
##      Vectorized version of Gauss Jordan        ##
####################################################

myGaussJordanVec <- function(A, m){
  
  # Perform Gauss Jordan elimination on A.
  # 
  # A: a square matrix.
  # m: the pivot element is A[m, m].
  # Returns a matrix with the identity matrix 
  # on the left and the inverse of A on the right.

  
  n <- dim(A)[1]
  B <- cbind(A, diag(rep(1,n)))
  
  for(k in 1:m)
  {
    B[k, ] <- B[k, ] / B[k,k]
    
    for(i in 1:n)
      if(i != k)
        B[i,] <- B[i,] - B[k, ] * B[i,k]
  }

  
  ## Function returns the matrix B
  return(B)
  
}



######################################################
##    Linear regression using Gauss Jordan          ##
######################################################

myLinearRegression <- function(X, Y){
  
  # Find the regression coefficient estimates beta_hat
  # corresponding to the model Y = X * beta + epsilon

  # X: an 'n row' by 'p column' matrix of input variables.
  # Y: an n-dimensional vector of responses
  
  n <- nrow(X)
  p <- ncol(X)
  
  Z <- cbind(rep(1,n),X, Y)
  A <- t(Z) %*% Z
  S <- myGaussJordanVec(A, p+1)

  
  beta_hat <- S[1:(p+1),p+2]
  
  
  ## Function returns the (p+1)-dimensional vector 
  ## beta_hat of regression coefficient estimates

  return(beta_hat)
  
}

########################################################
##                 TESTING                            ##
########################################################

testing_Linear_Regression <- function(){


  ## Define parameters
  n    <- 100
  p    <- 3

  ## Simulate data from our assumed model.
  ## We can assume that the true intercept is 0
  X    <- matrix(rnorm(n * p), nrow = n)
  beta <- matrix(1:p, nrow = p)

  Y    <- X %*% beta + rnorm(n)


  ## Save R's linear regression coefficients
  R_coef  <- coef(lm(Y ~ X))


  ## Save our linear regression coefficients
  my_coef <- myLinearRegression(X, Y)

  sum_square_diff <- sum((R_coef - my_coef)^2)
  
  if(sum_square_diff <= 0.001){
    return('Both results are identical')
  }else{
    return('There seems to be a problem...')
  }

}

