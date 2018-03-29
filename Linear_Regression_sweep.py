"""
 Description: This script implements linear regression 
 using the sweep operator.
"""
#from sklearn.linear_model import LinearRegression as lr
import numpy as np


################################
##      Sweep operator        ##
################################

def mySweep(A, m):
    
    """
    Perform a SWEEP operation on A with the pivot element A[m,m].
    
    :param A: a square matrix (np.array).
    :param m: the pivot element is A[m, m].
    :returns a swept matrix (np.array). Original matrix is unchanged.

    """
    
    B = np.copy(A)   
    n = B.shape[0]
    p = B.shape[1]

    for k in range(m):
        for i in range(n):
            for j in range(n):
                if i!=k and j!=k:
                    B[i,j] = B[i,j] - ((B[i,k]*B[k,j])/B[k,k])

        for i in range(n):
            if i!=k:
                B[i,k] = B[i,k] / B[k,k]

        for j in range(n):
            if j!=k:
                B[k,j] = B[k,j] / B[k,k]


        B[k,k] = -1/B[k,k]

    ## The function outputs the matrix (np.array) B
    return(B)
  



########################################################
##      Linear regression using Sweep operator        ##
########################################################

def myLinearRegression(X, Y):
  
  """
  Find the regression coefficient estimates beta_hat
  corresponding to the model Y = X * beta + epsilon

  X: an 'n row' by 'p column' matrix (np.array) of input variables.
  Y: an n-dimensional vector (np.array) of responses

  """

  n = X.shape[0]
  p = X.shape[1]

  Z = np.hstack((np.ones((n, 1)), X, Y.reshape(n, 1)))
  A = np.dot(np.transpose(Z), Z)
  S = mySweep(A, p + 1)

  ## Function returns the (p+1)-dimensional vector
  ## beta_hat of regression coefficient estimates
  beta_hat = S[range(p + 1), p + 1]
  return (beta_hat)


"""
########################################################
##                      TESTING                       ##
########################################################

def testing_Linear_Regression():

  n = 100
  p = 3

  X = np.random.random((n,p))
  beta = np.array(range(1,p+1))
  beta.shape = (p,1)

  epsilon = np.random.normal(0,1,n)
  epsilon.shape = (n,1)

  Y = np.matmul(X, beta) + epsilon

  regr = lr().fit(X,Y)

  python_coef =  np.append(regr.intercept_, regr.coef_)
  my_coef = myLinearRegression(X,Y)

  sum_square_diff = sum((python_coef - my_coef)**2)
  
  if(sum_square_diff <= 0.001):
    print('Both results are identical')
  else:
    print('There seems to be a problem...')



testing_Linear_Regression()
"""
