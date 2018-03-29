library(RcppArmadillo)
sourceCpp("Linear_Regression_sweep.cpp")

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
  my_coef <- myLinearRegressionC(X, Y)

  sum_square_diff <- sum((R_coef - my_coef)^2)

  if(sum_square_diff <= 0.001){
    return('Both results are identical')
  }else{
    return('There seems to be a problem...')
  }
  
}
