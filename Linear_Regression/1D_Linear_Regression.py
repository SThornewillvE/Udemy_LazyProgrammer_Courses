# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 12:34:21 2019

@author: sthornewillvonessen
"""

# Import Packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression


plt.style.use("seaborn")

# Define 

# Generate random data for use
X, y = make_regression(n_samples=100, n_features=1, noise = 50)


# Plot data to see what it looks like
plt.scatter(X, y)
plt.show()  # We can see that there is a slight correlation, but it is not very strong

def grad_desc(X, y):
    """
    Attempts to find the optimal values for a, b for the relationship between X and y for the equation `y = aX + b`
    using the gradient descent algorithm.
    
    :returns: a, b
    """
    
    
def linear_algebra(X, y):
    """
    Attempts to find the optimal values for a, b for the relationship between X and y for the equation `y = aX + b`
    using linear algebra.
    
    :returns: a, b
    """
    
    
def direct_calculation(X, y):
    """
    Attempts to find the optimal values for a, b for the relationship between X and y for the equation `y = aX + b`
    using equations for a and b.
    
    :returns: a, b
    """
    a = ((X*y).mean() - (X.mean()*y.mean()))/((X**2).mean()-(X.mean())**2)
    b = (y.mean()*(X**2).mean() - X.mean()*(X*y).mean())/((X**2).mean()-(X.mean())**2)
    
    return a, b

def plot_line(X, y, param_dict):
    """
    Plots straight line based on  X, y and parameters within the list.
    
    Note that the parameter list should be in the form `{label: [a, b]}`.
    
    Where `label` is the label to be shown in the plot.
    
    :returns:
    """
    
    dep_var = np.linspace(start=X.min(), stop=X.max(), num=200)
    dict_keys = list(param_dict.keys())
    
    plt.scatter(X, y)
    for key in dict_keys:
        a, b = param_dict[key]
        plt.plot(dep_var, (a*dep_var)+b, label=key)
    plt.legend()
    plt.show()
    
    return
    
    



# Calculate different values for a and b
a, b = direct_calculation(X, y)

# Create parameter dictionary
param_dict = {"direct_calc": [a, b]}

plot_line(X, y, param_dict)
















