/*
#########################################################
## Description: This script implements QR decomposition,
## linear regression, and eigen decomposition / PCA 
## based on QR.
#########################################################
*/ 

# include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

// [[Rcpp::export()]]
double signC(double d){
  return d<0?-1:d>0? 1:0;
}


/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
         QR decomposition 
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */  
  

// [[Rcpp::export()]]
List myQRC(const mat A){ 
  
  /*
  Perform QR decomposition on the matrix A
  Input: 
  A, an n x m matrix (mat)
 
  */ 
  int n = A.n_rows;
  int m = A.n_cols;
  
  mat R = mat(A);
  mat Q = eye(n, n);
  
  for(int k=0;k<m-1;k++){
    mat x = zeros(n, 1);
    
    for(int l=k;l<n;l++){
      x(l,0) = R(l,k);
    }
    
    mat v = mat(x);
    
    v(k) = x(k) + signC(x(k,0)) * norm(x, "fro");
    double s = norm(v, "fro");
    if(s != 0){
      mat u = v / s;
      R = R - 2 * (u * (u.t() * R));
      Q = Q - 2 * (u * (u.t() * Q));
    }
  }
  
  List output;
  
  
  // Function should output a List 'output', with 
  // Q.transpose and R
  // Q is an orthogonal n x n matrix
  // R is an upper triangular n x m matrix
  // Q and R satisfy the equation: A = Q %*% R
  output["Q"] = Q.t();
  output["R"] = R;
  return(output);
  

}
  
/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
         Linear regression using QR 
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  
  
// [[Rcpp::export()]]
mat myLinearRegressionC(const mat X, const mat Y){
    
  /*  
  Perform the linear regression of Y on X
  Input: 
  X is an n x p matrix of explanatory variables
  Y is an n dimensional vector of responses
 
  */  
  
  int n = X.n_rows;
  int p = X.n_cols;
  mat Z, R1, Y1, beta_ls;
  List result;
  Z = join_rows(ones<mat>(n,1),X);
  Z = join_rows(Z,Y);
  
  result = myQRC(Z);
  mat R = result["R"];
  
  R1 = R.submat(0,0,p,p);
  Y1 = R.submat(0,p+1,p,p+1);

  beta_ls = solve(R1, Y1);
  
  
  // Function returns the 'p+1' by '1' matrix 
  // beta_ls of regression coefficient estimates
  return(beta_ls.t());
  
}  

/* ~~~~~~~~~~~~~~~~~~~~~~~~ 
       PCA based on QR 
 ~~~~~~~~~~~~~~~~~~~~~~~~~~ */


// [[Rcpp::export()]]
List myEigen_QRC(const mat A, const int numIter = 1000){
  
  /*  
  
  Perform PCA on matrix A using your QR function, myQRC.
  Input:
  A: Square matrix
  numIter: Number of iterations
   
   */ 
  
  int r = A.n_rows;
  int c = A.n_cols;
  
  mat V = runif(r*r);
  mat Q;
  V.reshape(r,c);

  List output;
  
  for(int i=0; i < numIter; i++)
  {
    output = myQRC(V);
    mat Q = output["Q"];
    V = A.t() * Q;
    
  }

  output = myQRC(V);

  mat Q2 = output["Q"];
  mat R = output["R"];

  vec D = R.diag();
  //NumericMatrix D2 = wrap(D);
  

  // Function should output a list with D and V
  // D is a vector of eigenvalues of A
  // V is the matrix of eigenvectors of A (in the 
  // same order as the eigenvalues in D.)
  output["D"] = D;
  output["V"] = Q2;
  return(output);

}
  
