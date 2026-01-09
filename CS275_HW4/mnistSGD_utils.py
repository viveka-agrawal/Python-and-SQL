
# Imports
import numpy as np
import matplotlib.pyplot as plt
import time
import random 

def get_current_time_milliseconds():
    ''' 
        Get the current system time in milliseconds 

        Returns:
            Current time in milliseconds
    '''
    return int(round(time.time() * 1000))

def sigmoid(x):
    '''
        Computes the Sigmoid function

        Parameters:
            x: Array to compute sigmoid on

        Returns:
            sigmoid(x)
    '''
    return 1 / (1+np.exp(-x))

def logistic_loss_scaled(w, X, y):
    '''
        Compute the (negative) scaled logistic loss function

        Parameters:
            w : D vector of weights learned by logistic regression
            X : N x D matrix of observed data, each row is one example
            y : N x 1 vector of true class labels (-1 / +1)

        Returns:
            negative of the Scaled logistic loss
    '''
    # Some small epsilon
    eps = 1e-8

    # Add a dim to "w" so the dims are correct
    w = np.expand_dims(w, -1)

    # Compute the Logistic loss
    y_01 = (y+1)/2
    mu = sigmoid(np.matmul(X, w)).reshape(-1)
    mu[mu < eps] = eps # bound away from 0
    mu[mu > (1-eps)] = (1-eps) # bound away from 1
    nll = -np.sum((y_01 * np.log(mu)) + ((1.0-y_01) * np.log(1.0-mu)))

    # Scale the logistic loss
    N = X.shape[0]
    nll = nll / N
    return nll

def logistic_loss_scaled_grad(w, X, y):
    '''
        Compute the (negative) scaled logistic loss function gradient

        Parameters:
            w : D vector of weights learned by logistic regression
            X : N x D matrix of observed data, each row is one example
            y : N x 1 vector of true class labels (-1 / +1)

        Returns:
            negative of the Scaled logistic loss gradient
    '''
    # Some small epsilon
    eps = 1e-8

    # Add a dim to "w" so the dims are correct
    w = np.expand_dims(w, -1)

    # Compute the Logistic loss gradient
    y_01 = (y+1)/2
    mu = sigmoid(np.matmul(X, w)).reshape((-1,))
    mu[mu < eps] = eps # bound away from 0
    mu[mu > (1-eps)] = (1-eps) # bound away from 1
    grad = np.matmul(X.T, np.expand_dims((mu - y_01), 1))

    # Scale the logistic loss gradient
    N = X.shape[0]
    grad = grad / N

    grad = np.reshape(grad, [grad.shape[0],])

    return grad

def calc_log_reg_accuracy(w, X, y):
    '''
        Computes the accuracy metric

        Parameters:
            w : D x 1 vector of weights learned by logistic regression
            X : N x D matrix of observed data, each row is one example
            y : N x 1 vector of true class labels (-1 / +1)

        Returns:
            Accuracy
    '''

    yhat = sigmoid(np.matmul(X, w).squeeze())
    yhat = yhat > 0.5

    # Convert 0/1 labels to -1/+1
    yhat = 2*yhat - 1
    return np.sum(yhat == y) / y.shape[0]


def create_batches_from_full_dataset(X, y, num_batches):
    '''
        Batches the data into num_batches by evenly dividing the data into batches.

        Parameters:
            X : N x D matrix of observed data, each row is one example
            y : N x 1 vector of true class labels (-1 / +1)
            num_batches: The number of batches to create
        Returns:
            batches of data
    '''

    # shuffle the data idxs
    batch_idxs = list(range(X.shape[0]))
    random.shuffle(batch_idxs)

    # Compute the size of the batch
    batch_size = np.ceil(X.shape[0]/ num_batches)

    # Batch the data
    batches = []
    for b in range(num_batches):

        # Compute the start and endpoint of the batches
        s = int(b * batch_size)
        e = int(min(s+batch_size, X.shape[0]))

        # Extract the batch of data
        x_batch = X[batch_idxs[s:e]]
        y_batch = y[batch_idxs[s:e]]
        batches.append((x_batch, y_batch))

    return batches
