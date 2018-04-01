#########################################################
## Description: This script implements logistic regression
## using iterated reweighted least squares using the code 
## we have written for linear regression based on QR 
## decomposition
#########################################################


## library(Rcpp)
## sourceCpp(name_of_cpp_file)

##################################
##  QR decomposition            ##
##################################


myQR <- function(A){
  
  ## Perform QR decomposition on the matrix A
  ## Input: 
  ## A, an n x m matrix

  
  n = dim(A)[1]
  m = dim(A)[2]
  
  R = A
  Q = diag(n)
  
  for(k in 1 : (m - 1)){
    x = matrix(0, n, 1)
    x[k:n, 1] = R[k:n, k]
    v = x
    
    v[k] = x[k] + sign(x[k,1]) * norm(x, type="F")
    s = norm(v, "F")
    
    if(s != 0){
      u = v / s
      R = R - 2 * (u %*% (t(u) %*% R))
      Q = Q - 2 * (u %*% (t(u) %*% Q))
    }
  }
  
  
  ## Function should output a list with Q.transpose and R
  ## Q is an orthogonal n x n matrix
  ## R is an upper triangular n x m matrix
  ## Q and R satisfy the equation: A = Q %*% R
  return(list("Q" = t(Q), "R" = R))
  
}

###############################################
##   Linear regression based on QR           ##
###############################################

myLM <- function(X, Y){
  
  ## Perform the linear regression of Y on X
  ## Input: 
  ## X is an n x p matrix of explanatory variables
  ## Y is an n dimensional vector of responses

  n=nrow(X)
  p=ncol(X)
  Z = cbind(X, Y)
  R = myQR(Z)$R
  R1 = R[1:(p), 1:(p)]
  Y1 = R[1:(p), p+1]
  
  beta_ls = solve(R1, Y1)
  
  
  
  ## Function returns the least squares solution vector
  return(beta_ls)
  
}

######################################
##      Logistic regression         ##
######################################

## Expit/sigmoid function
expit <- function(x){
  1 / (1 + exp(-x))
}

myLogistic <- function(X, Y){
  
  ## Perform the logistic regression of Y on X
  ## Input: 
  ## X is an n x p matrix of explanatory variables
  ## Y is an n dimensional vector of binary responses

  n = dim(X)[1]
  p = dim(X)[2]
  
  beta = matrix(rep(0,p), nrow = p)
  
  epsilon = 1e-6
  
  repeat
  {
    X_beta_product = X %*% beta
    pr = expit(X_beta_product)
    z = X_beta_product + (Y - pr)/(pr*(1-pr))
    sqw = sqrt(pr*(1-pr))
    mw = matrix(sqw, n, p)
    xw = mw*X
    yw = sqw*z
    
    beta_n = myLM(xw, yw)
    error = sum(abs(beta_n - beta))
    beta = beta_n
    if (error < epsilon)
      break
    
    
    
  }
  
  
  
  
  ## Function returns the logistic regression solution vector
  return (beta)  
  
}


# Testing
# n <- 100
# p <- 5
# 
# X    <- matrix(rnorm(n * p), nrow = n)
# beta <- rnorm(p)
# Y    <- 1 * (runif(n) < expit(X %*% beta))
# 
# logistic_beta <- myLogistic(X, Y)
# logistic_R = glm(formula = Y ~ X + 0,  family=binomial("logit"))
# 
# sum_square_diff = sum((logistic_beta-logistic_R$coefficients)^2)
# 
# if(sum_square_diff <= 0.001){
#     print('Logistic Regression results are identical!!')
#   }else{
#     print('There seems to be a problem...')
#   }


