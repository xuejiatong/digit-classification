import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import warnings

def train_one_vs_all(X, y, num_classes, lambda_val):
    '''
    Train a one vs. all logistic regression
    
    Inputs: 
      X                data matrix (2d array shape m x n)
      y                label vector with entries from 0 to 
                       num_classes - 1 (1d array length m)
      num_classes      number of classes (integer)
      lambda_val       regularization parameter (scalar)

    Outputs:
      weight_vectors   matrix of weight vectors for each class 
                       weight vector for class c in the cth column
                       (2d array shape n x num_classes)
      intercepts       vector of intercepts for all classes
                       (1d array length num_classes)                       
            
    '''
    
    # initialize the weight vectors, intercepts, step size, and iterations
    intercepts = np.zeros(num_classes)
    weight_vectors = np.zeros((X.shape[1], num_classes))
    alpha = 0.01 # step size (scalar)
    iters = 500 # number of iterations (integer)
    
    # iterate for each class
    for num in range(num_classes):        
        # get y for each class
        i = np.ones(X.shape[0])*num
        y_tmp = (y == i).astype(int)
        
        # regularized gradient descent
        weight_vector, intercept = train_logistic_regression(X, y_tmp, lambda_val)
        
        intercepts[num] = intercept
        weight_vectors[:,num] = np.transpose(weight_vector)
    
    return weight_vectors, intercepts


def predict_one_vs_all(X, weight_vectors, intercepts):
    '''
    Train a one vs. all logistic regression
    
    Inputs: 
      X                data matrix (2d array shape m x n)
      weight_vectors   matrix of weight vectors for each class 
                       weight vector for class c in the cth column
                       (2d array shape n x num_classes)
      intercepts       vector of intercepts for all classes
                       (1d array length num_classes)   
                       
    Outputs:
      predictions      vector of predictions for examples in X
                       (1d array length m)            
    '''    
    
    # Write code here
    
    # Hint: use a matrix vector multiplication to simultaneously make
    # predictions for all classes. Don't forget to add the intercept values
    
    z = np.dot(X, weight_vectors)
    num_classes = intercepts.shape[0]
    for num in range(num_classes):
        z[:,num] += intercepts[num]
    
    predictions = np.argmax(z, axis = 1)
    
    # Hint: look up the np.argmax function. It can find the index of
    # the largest value in an array, or in each row/column of an array
    
    return predictions


def train_logistic_regression(X, y, lambda_val):
    '''
    Train a regularized logistic regression model
    
    Inputs:
      X           data matrix (2d array shape m x n)
      y           label vector with 0/1 entries (1d array length m)
      lambda_val  regularization parameter (scalar)

    Outputs:
      weights     weight vector (1d array length n)
      intercept   intercept parameter (scalar)
    '''
    model = linear_model.LogisticRegression(C=2./lambda_val, solver='lbfgs')

    # call model.fit(X, y) while suppressing warnings about convergence
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)

    weight_vector = model.coef_.ravel()
    intercept = model.intercept_
    return weight_vector, intercept

def display_data(X, im_width=None, return_mosaic=False):
    '''
    Display data rows as mosaic image
    '''
    
    m, n = X.shape
    
    if im_width is None:
        im_width = np.int(np.sqrt(n).astype('int'))
    
    im_height = np.int(n / im_width)
    
    if not im_width * im_height == n:
        raise ValueError('cannot determine image dimensions')

    X = X / (2*np.max(np.abs(X), axis=1, keepdims=True)) + 0.5

    # Compute rows, cols
    display_rows = np.floor(np.sqrt(m))
    display_cols = np.ceil( m / display_rows )

    display_rows = display_rows.astype('int')
    display_cols = display_cols.astype('int')
    
    fig = plt.figure(1, (6., 6.))
    
    # convert each row to image
    images = [X[i,:].reshape([im_height, im_width])  for i in range(m)]

    # pad images for nice display
    pad = 1
    images = [np.lib.pad(images[i], (pad,0), 'constant') for i in range(m)]

    # Assemble the image into a mosaic
    rows = []
    for i in range(display_rows):
        row_start =   i   * display_cols
        row_end   = (i+1) * display_cols

        im  = np.concatenate( images[row_start:row_end], axis=1 )

        # Build the row first as an array of the correct size
        row = np.zeros( (im_height + pad, (im_width + pad)*display_cols))
        h,w = im.shape

        # Now populate it with the image
        row[:h, :w] = im
        rows.append(row)

    # Concatenate rows to get the final result
    mosaic = np.concatenate(rows,  axis=0)

    plt.imshow(mosaic, cmap='gray', clim=[0,1])
    plt.axis('off')
    plt.show()

    if return_mosaic:
        return mosaic
    else:
        return