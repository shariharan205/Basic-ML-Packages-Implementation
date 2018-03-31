#########################################################
## Description: This script implements QR decomposition,
## linear regression, and eigen decomposition / PCA 
## based on QR.
#########################################################

# library(Rcpp)
# filename <- 'QR.cpp'
# sourceCpp(filename)

##################################
## Function 1: QR decomposition ##
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
##        Linear regression based on QR      ##
###############################################

myLM <- function(X, Y){
  
  ## Perform the linear regression of Y on X
  ## Input: 
  ## X is an n x p matrix of explanatory variables
  ## Y is an n dimensional vector of responses

  n = dim(X)[1]
  p = dim(X)[2]
  Z = cbind(matrix(1, n, 1), X, matrix(Y, n, 1))
  
  result = myQR(Z)
  R = result$R
  
  R1 = R[1:(p+1),1:(p+1)]
  Y1 = R[1:(p+1), p+2]
  beta_ls = solve(R1, Y1)
  
  
  ## Function returns the 1 x (p + 1) vector beta_ls, 
  ## the least squares solution vector
  return(beta_ls)
  
}

##################################
##   PCA based on QR            ##
##################################

myEigen_QR <- function(A, numIter = 1000){
  
  ## Perform PCA on matrix A using your QR function, myQRC.
  ## Input:
  ## A: Square matrix
  ## numIter: Number of iterations

  r = dim(A)[1]
  c = dim(A)[2]
  
  V = matrix(runif(r * r), nrow = r)
  

  for(i in 1 : numIter){
    output = myQR(V)
    Q = output$Q
    V = t(A) %*% Q
  }
  
  output = myQRC(V)
  V = output$Q
  R = output$R
  D = diag(R)

  
  ## Function should output a list with D and V
  ## D is a vector of eigenvalues of A
  ## V is the matrix of eigenvectors of A (in the 
  ## same order as the eigenvalues in D.)
  return(list("D" = D, "V" = Q))
  
}



########################################################
##                 TESTING                            ##
########################################################

# testing_QR <- function(){
# 
#   
#   n    <- 100
#   p    <- 3
# 
#   X    <- matrix(rnorm(n * p), nrow = n)
# 
#   my_qr = myQR(X)
#   my_qrc = myQRC(X)
# 
#   qr_X = my_qr$Q %*% my_qr$R
#   qr_X_c = my_qrc$Q %*% my_qrc$R
#   
#   
# 
#   sum_square_diff <- sum((X - qr_X)^2)
#   sum_square_diff_c <- sum((X - qr_X_c)^2)
#   
#   if(sum_square_diff <= 0.001){
#       print('QR Decomposition results of R are identical')
#     }else{
#       print('There seems to be a problem with QR Decomposition in R')
#     }
#   
#   if(sum_square_diff_c <= 0.001){
#     print('QR Decomposition results of Rcpp are identical')
#   }else{
#     print('There seems to be a problem with QR Decomposition in RCpp')
#   }
#   
# 
#   beta <- matrix(1:p, nrow = p)
#   Y    <- X %*% beta + rnorm(n)
# 
#   ## Save R's linear regression coefficients
#   R_coef  <- coef(lm(Y ~ X))
# 
#   ## Save our linear regression coefficients
#   my_coef <- myLM(X, Y)
#   my_coef_c <- myLinearRegressionC(X,Y);
# 
#   sum_square_diff <- sum((R_coef - my_coef)^2);
#   sum_square_diff_c <- sum((R_coef - my_coef_c)^2);
#   
#   if(sum_square_diff <= 0.001){
#     print('Linear Regression results in R are identical')
#   }else{
#     print('There seems to be a problem with Linear Regression in R')
#   }
#   
#   if(sum_square_diff_c <= 0.001){
#     print('Linear Regression results in RCpp are identical')
#   }else{
#     print('There seems to be a problem with Linear Regression in RCpp')
#   }
#   
#   my_eigen_qr <- myEigen_QR(t(X) %*% X)
#   my_eigen_qr_values <- my_eigen_qr$D
#   #my_eigen_qr_vectors <- my_eigen_qr$V
#   my_eigen_qr_c_values <- myEigen_QRC(t(X) %*% X)["D"]
# 
#   
#   r_eigen <- as.list(eigen(t(X) %*% X))
#   r_eigen_values <- r_eigen$values
#   #r_eigen_vectors <- r_eigen$vectors
#   D <- myEigen_QRC(t(X) %*% X)$D
#   V <- myEigen_QRC(t(X) %*% X)$V
#   print(dim(D))
#   print(dim(V))
#   print(class(D))
#   print(class(V))
#   
#   print(r_eigen_values)
#   print(r_eigen$vectors)
# 
#   
# 
#   sum_square_diff_r_values <- sum((my_eigen_qr_values - r_eigen_values)^2);
#   #sum_square_diff_r_vectors <- sum((my_eigen_qr_vectors - r_eigen_vectors)^2);
#   sum_square_diff_c_values <- sum((unlist(my_eigen_qr_c_values) - r_eigen_values)^2);
# 
#   
#   if(sum_square_diff_r_values <= 0.001){
#     print('PCA results in R are identical ')
#   }else{
#     print('There seems to be a problem with PCA in R')
#   }
# 
#   
#   if(sum_square_diff_c_values <= 0.001){
#     print('PCA results in RCpp are identical ')
#   }else{
#     print('There seems to be a problem with PCA in RCpp')
#   }
#   
# }

