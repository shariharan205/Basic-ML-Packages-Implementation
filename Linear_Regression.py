"""
This script implements linear regression using Gauss-Jordan elimination in both plain and
vectorized forms.
"""
from sklearn.linear_model import LinearRegression as lr
import numpy as np


###############################################
## Plain version of Gauss Jordan             ##
###############################################


def myGaussJordan(A, m):
    """
    Perform Gauss Jordan elimination on A.

    A: a square matrix.
    m: the pivot element is A[m, m].
    Returns a matrix with the identity matrix
    on the left and the inverse of A on the right.
    """

    n = A.shape[0]
    B = np.hstack((A, np.identity(n)))

    for k in range(m):
        a = B[k, k]

        for j in range(n * 2):
            B[k, j] = B[k, j] / float(a)

        for i in range(n):
            if i != k:
                b = B[i, k]
                for j in range(n * 2):
                    B[i, j] = B[i, j] - b * B[k, j]

    ## Function returns the np.array B
    return B


####################################################
## Vectorized version of Gauss Jordan             ##
####################################################

def myGaussJordanVec(A, m):
    """
    Perform Gauss Jordan elimination on A.

    A: a square matrix.
    m: the pivot element is A[m, m].
    Returns a matrix with the identity matrix
    on the left and the inverse of A on the right.
    """

    n = A.shape[0]
    B = np.hstack((A, np.identity(n)))

    for k in range(m):

        B[k, :] = B[k, :] / B[k, k]

        for i in range(n):
            if i != k:
                B[i, :] = B[i, :] - B[k, :] * B[i, k]

    ## Function returns the np.array B
    return B


######################################################
##   Linear regression using Gauss Jordan           ##
######################################################

def myLinearRegression(X, Y):
    """
    Find the regression coefficient estimates beta_hat
    corresponding to the model Y = X * beta + epsilon

    X: an 'n row' by 'p column' matrix (np.array) of input variables.
    Y: an n-dimensional vector (np.array) of responses

    """
    n = X.shape[0]
    p = X.shape[1]

    Z = np.hstack((np.ones((n, 1)), X, Y))
    A = np.matmul(np.transpose(Z), Z)
    S = myGaussJordanVec(A, p + 1)

    beta_hat = S[0:(p + 1), p + 1]
    beta_hat.shape = (p + 1)

    ## Function returns the (p+1)-dimensional vector (np.array)
    ## beta_hat of regression coefficient estimates
    return beta_hat


########################################################
##                  TESTING                           ##
########################################################

def testing_Linear_Regression():
    n = 100
    p = 3

    X = np.random.random((n, p))
    beta = np.array(range(1, p + 1))
    beta.shape = (p, 1)

    epsilon = np.random.normal(0, 1, n)
    epsilon.shape = (n, 1)

    Y = np.matmul(X, beta) + epsilon

    regr = lr().fit(X, Y)

    python_coef = np.append(regr.intercept_, regr.coef_)
    my_coef = myLinearRegression(X, Y)

    sum_square_diff = sum((python_coef - my_coef) ** 2)

    if sum_square_diff <= 0.001:
        print('Both results are identical')

    else:
        print('There seems to be a problem...')


testing_Linear_Regression()
