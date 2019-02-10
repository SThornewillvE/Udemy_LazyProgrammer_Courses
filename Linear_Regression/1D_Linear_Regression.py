# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 12:34:21 2019

@author: sthornewillvonessen
"""

# Import Packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Custom packages
from R_sq import R_sq_eval

plt.style.use("seaborn")

# Define 

# Generate random data for use
X, y = make_regression(n_samples=100, n_features=1, noise = 50)

# Plot data to see what it looks like
plt.scatter(X, y)
plt.show()  # We can see that there is a slight correlation, but it is not very strong

def grad_desc(X, y, alpha=0.1, steps=1000):
    """
    Attempts to find the optimal values for a, b for the relationship between X and y for the equation `y = aX + b`
    using the gradient descent algorithm.
    
    Note that alpha is equal to the learning rate and steps are the number of steps in the descent.
    
    :returns: a, b
    """
    # Check that x_i and y_i are the same length
    if n != len(X):
        print("Error, X and y do not contain an equal number of observations")
        return
    
    # Initialize random values for a and b
    a = np.random.randint(0, 100)
    b = np.random.randint(0, 100)
    
    for i in range(steps):
        # Find loss
        
        d_a = ((y-a*X[:, 0]-b)*-2*X[:, 0]/(len(y))).sum()
        d_b = ((y-a*X[:, 0]-b)*-2/(len(y))).sum()
         
        # Update values
        a -= alpha * d_a
        b -= alpha * d_b
    
    return a, b
    
    
def linear_algebra(X, y):
    """
    Fits a linear regression to data to predict the experimental outcome.
    
    x_i: Input, numpy array, list or series.
    y_i: Observed output, 
    
    Note: x_i and y_i must be equal.
    
    Returns the predicted/estimated slope and intercept of the regression in the form of a list:
    
    list: [slope, intercept]
    """
    
    # Convert to np array
    x_i = np.array(X)
    y_i = np.array(y)
    
    # First we'll calculate n
    n = len(x_i)
    
    # Check that x_i and y_i are the same length
    if n != len(X):
        print("Error, X and y do not contain an equal number of observations")
        return
    
    # Calculate the sums as required
    sum_x = x_i.sum()
    sum_y = y_i.sum()
    sum_xy = x_i.dot(y_i)
    sum_xx = sum(x_i**2)
    
    # Solve for Ax=b => x=A^{-1}b
    
    # Assemble Matrix A
    A = np.matrix("{}, {}; {}, {}".format(n, sum_x, sum_x, sum_xx))
    
    # Calculate A^-1
    try: A_inv = A.I
    except: 
        print("Matrix has no inverse")
        return
    
    # Assemble vector b
    b = np.matrix("{}; {}".format(sum_y, sum_xy))
    
    # Calculate x
    x = A_inv * b
    
    # Convert formats
    slope = float(x[1])
    intercept = float(x[0])
    
    # Return as DataFrame
    return slope, intercept


def plot_line(X, y, param_dict):
    """
    Plots straight line based on  X, y and parameters within the list.
    
    Note that the parameter list should be in the form `{label: [a, b]}`.
    
    Where `label` is the label to be shown in the plot.
    
    :returns:
    """
    
    dep_var = np.linspace(start=X.min(), stop=X.max(), num=len(y))
    dict_keys = list(param_dict.keys())
    
    plt.scatter(X, y, alpha = 0.7)
    for key in dict_keys:
        a, b = param_dict[key]
        y_pred = (a*X[:, 0])+b
        r2 = R_sq_eval(y, y_pred)
        
        if key == 'sklearn':
            r2 = r2_score(y, y_pred)
        
        plt.plot(dep_var, (a*dep_var + b), label="{}, r sq; {}".format(key, r2), alpha = 0.8)
    plt.legend()
    plt.show()
    
    return

lm = LinearRegression()    
lm.fit(X, y)

# Create parameter dictionary
param_dict = {"gradient_descent": list(grad_desc(X, y)),
              "linear_algebra": list(linear_algebra(X[:, 0], y)),
              "sklearn": [lm.coef_[0], lm.intercept_],
              "average": [0, y.mean()]}

plot_line(X, y, param_dict)



















