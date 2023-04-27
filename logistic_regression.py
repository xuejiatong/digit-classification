import numpy as np

def logistic(z):
    """
    The logistic function
    Input:
       z   numpy array (any shape)
    Output:
       p   numpy array with same shape as z, where p = logistic(z) entrywise
    """
    
    p = 1/(1+np.exp(-z))
    return p

def cost_function(X, y, theta):
    """
    Compute the cost function for a particular data set and hypothesis (weight vector)
    Inputs:
        X      data matrix (2d numpy array with shape m x n)
        y      label vector (1d numpy array -- length m)
        theta  parameter vector (1d numpy array -- length n)
    Output:
        cost   the value of the cost function (scalar)
    """
    
    # REPLACE CODE BELOW WITH CORRECT CODE
    theta_x = np.dot(X, np.transpose(theta))
    h = logistic(theta_x)
    ones = np.ones(X.shape[0])
    cost = -np.dot(y, np.log(h)) - np.dot((ones-y), np.log(ones-h))
    return cost

def gradient_descent( X, y, theta, alpha, iters ):
    """
    Fit a logistic regression model by gradient descent.
    Inputs:
        X          data matrix (2d numpy array with shape m x n)
        y          label vector (1d numpy array -- length m)
        theta      initial parameter vector (1d numpy array -- length n)
        alpha      step size (scalar)
        iters      number of iterations (integer)
    Return (tuple):
        theta      learned parameter vector (1d numpy array -- length n)
        J_history  cost function in iteration (1d numpy array -- length iters)
    """

    # REPLACE CODE BELOW WITH CORRECT CODE
    d_theta = theta
    J_history = np.zeros(iters)
    for i in range(0, iters):
        J_history[i] = cost_function(X, y, d_theta)
        h = logistic(np.dot(X, np.transpose(d_theta)))
        der = np.dot(np.transpose(X), (h-y))
        d_theta = d_theta - der*alpha
    return d_theta, J_history