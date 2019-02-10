# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 12:26:32 2019

@author: sthornewillvonessen
"""

# Import Packages
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Custom packages
from R_sq import R_sq_eval

plt.style.use("seaborn")

# Generate random data for use
X, y = make_regression(n_samples=100, n_features=2, noise = 0)

# Plot data to see what it looks like
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y)
ax.show()  # We can see that there is a slight correlation, but it is not very strong

def create_const(X):
    """
    Takes a numpy array of shape nxm and adds a column of ones. Hence, returns shape nx(m+1)
    
    :returns: X_new
    """
    
    X_new = np.zeros((X.shape[0], X.shape[1]+1))
    X_new[:, :2] = X
    X_new[:, 2] = 1
    
    return X_new
    

def grad_desc(X, y, alpha=0.0001, steps=1000):
    """
    Attempts to find the optimal values Theta for the relationship between X and y for the equation `y = X*Theta`
    using the gradient descent algorithm.
    
    Note that alpha is equal to the learning rate and steps are the number of steps in the descent.
    
    :returns: a, b
    """
    
    # First we'll calculate n
    n = len(y)
    
    # Check that x_i and y_i are the same length
    if n != len(X):
        print("Error, X and y do not contain an equal number of observations")
        return
    
    # Initialize array for Theta
    Theta = np.array([100, 100, 100])
    
    for i in range(steps):
        
        # Find prediction
        y_pred = np.matmul(X, Theta)
        
        # Find loss
        loss = y_pred - y 
        
        # Update weights
        Theta = Theta - (alpha * np.matmul(X.T, loss))  # X.T * loss is the gradient
    
    return Theta
    
    
def linear_algebra(X, y):
    """
    Attempts to find the optimal values Theta for the relationship between X and y for the equation `y = X*Theta`
    using the gradient descent algorithm.
    
    Note that X and y must have the same number of observations.
    
    Returns the predicted/estimated slope and intercept of the regression in the form of a list:
    
    :returns
    """
    
    # First we'll calculate n
    n = len(y)
    
    # Check that x_i and y_i are the same length
    if n != len(X):
        print("Error, X and y do not contain an equal number of observations")
        return
    
    Xt_X = np.matmul(X.T, X)
    Xt_y = np.matmul(X.T, y)
    
    Theta = np.linalg.solve(Xt_X, Xt_y)

    return Theta


def plot_line(X, y, param_dict):
    """
    Plots straight line based on  X, y and parameters within the list.
    
    Note that the parameter list should be in the form `{label: [a, b]}`.
    
    Where `label` is the label to be shown in the plot.
    
    :returns:
    """
    
    # Create linear space for indepdep_var
    indep_var = np.zeros((X.shape[0], X.shape[1]))
    
    for i in range(indep_var.shape[1]):
        if i == indep_var.shape[1]-1:
            indep_var[:, i] = 1
            continue
        indep_var[:, i] = np.linspace(start=X[:, i].min(), stop=X[:, i].max(), num=len(X))
    
    dict_keys = list(param_dict.keys())
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for key in dict_keys:
        params = param_dict[key]
        y_pred = np.matmul(X, params)
        r2 = R_sq_eval(y, y_pred)
        
        if key == 'sklearn':
            r2 = r2_score(y, y_pred)
        
        ## TODO Fix plot
        ax.plot(indep_var[:, 0], indep_var[:, 1], np.matmul(indep_var, params), label="{}, r sq; {}".format(key, r2), alpha = 0.8)
        ax.show()
    
    return

# Create constant for X
X = create_const(X)

# Create parameter dictionary
param_dict = {"linear_algebra": linear_algebra(X, y),
              "average": [0, 0, y.mean()]}

plot_line(X, y, param_dict)