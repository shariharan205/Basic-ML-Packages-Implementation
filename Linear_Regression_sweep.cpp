/*
####################################################
## Description: This script implements linear regression
## using the sweep operator
####################################################
*/


# include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;



/* ~~~~~~~~~~~~~~~~~~~~~~~~~
          Sweep operator
   ~~~~~~~~~~~~~~~~~~~~~~~~~ */

// [[Rcpp::export()]]
mat mySweepC(const mat A, int m){

  /*
  Perform a SWEEP operation on A with the pivot element A[m,m].

  A: a square matrix (mat).
  m: the pivot element is A[m, m].
  Returns a swept matrix B (which is m by m).

  */

  mat B = A;
  int n = B.n_rows;

  int i,k,j;

  for(k=0; k<m; k++)
  {
      for(i=0; i<n; i++)
      {
          for(j=0; j<n; j++)
          {
              if(i!=k && j!=k)
                B(i,j) = B(i,j) - ((B(i,k)*B(k,j)/B(k,k)));
          }

      }

      for(i=0; i<n; i++)
      {
          if(i!=k)
            B(i,k) = B(i,k)/B(k,k);
      }
      
      for(j=0; j<n; j++)
      {
        if(j!=k)
          B(k,j) = B(k,j)/B(k,k);
      }

      B(k,k) = -1/B(k,k);
  }

  // Return swept matrix B
  return(B);

}


/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      Linear regression using the sweep operator
 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */


// [[Rcpp::export()]]
mat myLinearRegressionC(const mat X, const mat Y){

  /*
  Find the regression coefficient estimates beta_hat
  corresponding to the model Y = X * beta + epsilon

  X: an 'n row' by 'p column' matrix of input variables.
  Y: an 'n row' by '1 column' matrix of responses

  */

  int n = X.n_rows;
  int p = X.n_cols;

  mat Z,A,S, beta_hat;

  Z = join_rows(ones<mat>(n,1),X);
  Z = join_rows(Z,Y);

  A = Z.t() * Z;
  S = mySweepC(A, p+1);

  beta_hat = S.submat(0,p+1,p,p+1);

  // Function returns the 'p+1' by '1' matrix
  // beta_hat of regression coefficient estimates
  return(beta_hat);

}



// // R code blocks can be included in C++ files processed with sourceCpp
// // (useful for testing and development). The R code will be automatically 
// // run after the compilation.
// //
// 
// /*** R
// testing_Linear_RegressionC <- function(){
//   
//   
// ## Define parameters
//   n    <- 100
//   p    <- 5
//   
// ## We can assume that the true intercept is 0
//   X    <- matrix(rnorm(n * p), nrow = n)
//   
//   beta <- matrix(1:p, nrow = p)
//   Y    <- X %*% beta + rnorm(n)
//   
//   
// ## Save R's linear regression coefficients
//   R_coef  <- coef(lm(Y ~ X))
// #print(R_coef)
//   
// ## Save our linear regression coefficients
//   my_coef <- myLinearRegressionC(X, Y)
// #print(my_coef)
//   sum_square_diff <- sum((R_coef - my_coef)^2)
// #print(sum_square_diff)
//   
//   if(sum_square_diff <= 0.001){
//     return('Both results are identical')
//   }else{
//     return('There seems to be a problem...')
//   }
//   
// }
// */