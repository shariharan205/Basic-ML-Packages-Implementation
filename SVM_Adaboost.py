#########################################################
## Description: This script implements a support vector machine, an adaboost classifier
#########################################################

import numpy as np
import sklearn.datasets as ds
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def prepare_data(valid_digits=np.array((6, 5))):
    ## valid_digits is a vector containing the digits
    ## we wish to classify.
    ## Do not change anything inside of this function
    if len(valid_digits) != 2:
        raise Exception("Error: you must specify exactly 2 digits for classification!")

    data = ds.load_digits()
    labels = data['target']
    features = data['data']

    X = features[(labels == valid_digits[0]) | (labels == valid_digits[1]), :]
    Y = labels[(labels == valid_digits[0]) | (labels == valid_digits[1]),]

    X = np.asarray(map(lambda k: X[k, :] / X[k, :].max(), range(0, len(X))))

    Y[Y == valid_digits[0]] = 0
    Y[Y == valid_digits[1]] = 1

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=10)
    Y_train = Y_train.reshape((len(Y_train), 1))
    Y_test = Y_test.reshape((len(Y_test), 1))

    return X_train, Y_train, X_test, Y_test


####################################################
##           Support vector machine               ##
####################################################

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## Train an SVM to classify the digits data ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

def my_SVM(X_train, Y_train, X_test, Y_test, lamb=0.01, num_iterations=1000, learning_rate=0.1):
    ## X_train: Training set of features
    ## Y_train: Training set of labels corresponding to X_train
    ## X_test: Testing set of features
    ## Y_test: Testing set of labels correspdonding to X_test
    ## lamb: Regularization parameter
    ## num_iterations: Number of iterations.
    ## learning_rate: Learning rate.

    ## Function should learn the parameters of an SVM.


    n = X_train.shape[0]
    p = X_train.shape[1] + 1
    X_train1 = np.concatenate((np.repeat(1, n, axis=0).reshape((n, 1)), X_train), axis=1)
    Y_train = 2 * Y_train - 1
    beta = np.repeat(0., p, axis=0).reshape((p, 1))

    ntest = X_test.shape[0]
    X_test1 = np.concatenate((np.repeat(1, ntest, axis=0).reshape((ntest, 1)), X_test), axis=1)
    Y_test = 2 * Y_test - 1

    acc_train = np.repeat(0., num_iterations, axis=0)
    acc_test = np.repeat(0., num_iterations, axis=0)

    for iter in xrange(num_iterations):
        sc = X_train1.dot(beta)
        db = (sc * Y_train) < 1
        dbeta = X_train1.T.dot(db * Y_train) / n
        beta = beta + learning_rate * dbeta - lamb * beta

        # accuracy on training and testing data
        acc_train[iter] = np.mean(np.sign(sc) == Y_train)
        acc_test[iter] = np.mean(np.sign(X_test1.dot(beta)) == Y_test)

        if iter % 50 == 0:
            print("Iteration ", iter, "Training Accuracy: ", acc_train[iter], "Testing Accuracy: ", acc_test[iter])

    ## Function outputs 3 things:
    ## 1. The learned parameters of the SVM, beta
    ## 2. The accuracy over the training set, acc_train (a "num_iterations" dimensional vector).
    ## 3. The accuracy over the testing set, acc_test (a "num_iterations" dimensional vector).

    return beta, acc_train, acc_test


######################################
##        Adaboost                  ##
######################################

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## Use Adaboost to classify the digits data ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

def my_Adaboost(X_train, Y_train, X_test, Y_test, num_iterations=1000):
    ## X_train: Training set of features
    ## Y_train: Training set of labels corresponding to X_train
    ## X_test: Testing set of features
    ## Y_test: Testing set of labels correspdonding to X_test
    ## num_iterations: Number of iterations.

    ## Function should learn the parameters of an Adaboost classifier.

    n = X_train.shape[0]
    p = X_train.shape[1]
    threshold = 0.8

    X_train1 = 2 * (X_train > threshold) - 1
    Y_train = 2 * Y_train - 1

    X_test1 = 2 * (X_test > threshold) - 1
    Y_test = 2 * Y_test - 1

    beta = np.repeat(0., p).reshape((p, 1))
    w = np.repeat(1. / n, n).reshape((n, 1))

    weak_results = np.multiply(Y_train, X_train1) > 0

    acc_train = np.repeat(0., num_iterations, axis=0)
    acc_test = np.repeat(0., num_iterations, axis=0)

    for iter in xrange(num_iterations):
        w = w / np.sum(w)

        wt_weak_results = w * weak_results
        acc_wt = np.sum(wt_weak_results, axis=0)
        error = 1 - acc_wt

        j = np.argmin(error)
        dbeta = np.log((1 - error[j]) / error[j]) / 2
        beta[j] = beta[j] + dbeta

        w = w * np.exp(-dbeta * weak_results[:, j].reshape((n, 1)))

        sc = X_train1.dot(beta)
        acc_train[iter] = np.mean(np.sign(sc) == Y_train)
        acc_test[iter] = np.mean(np.sign(X_test1.dot(beta)) == Y_test)

        if iter % 50 == 0:
            print("Iteration ", iter, "Training Accuracy: ", acc_train[iter], "Testing Accuracy: ", acc_test[iter])

    ## Function outputs 3 things:
    ## 1. The learned parameters of the adaboost classifier, beta
    ## 2. The accuracy over the training set, acc_train (a "num_iterations" dimensional vector).
    ## 3. The accuracy over the testing set, acc_test (a "num_iterations" dimensional vector).
    return beta, acc_train, acc_test


########################
##        TESTING     ##
########################
"""
import time

start = time.time()
beta, acc_train, acc_test = my_Adaboost(X_train, Y_train, X_test, Y_test)
end = time.time()
print "Time taken to run Adaboost for 1000 iterations in seconds:", end - start
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.ylabel('Training Accuracy - Adaboost')
plt.xlabel('Iteration')
plt.plot(range(len(acc_train)), acc_train)
plt.subplot(1, 2, 2)
plt.ylabel('Testing Accuracy - Adaboost')
plt.xlabel('Iteration')
plt.plot(range(len(acc_test)), acc_test)
plt.show()

"""

"""
beta, acc_train, acc_test = my_SVM(X_train, Y_train, X_test, Y_test, 0.01, 100, 0.1)
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.ylabel('Training Accuracy - SVM')
plt.xlabel('Iteration')
plt.plot(range(len(acc_train)), acc_train)
plt.subplot(1, 2, 2)
plt.ylabel('Testing Accuracy - SVM')
plt.xlabel('Iteration')
plt.plot(range(len(acc_test)), acc_test)
plt.show()
"""
