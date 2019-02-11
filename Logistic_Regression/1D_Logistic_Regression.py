# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 09:10:49 2019

@author: sthornewillvonessen
"""

# Import Packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

plt.style.use("seaborn")

# Generate random data for use
X, y = make_classification(n_samples=100, n_features=1, 
                           n_informative=1, n_redundant=0, 
                           n_classes=2, n_clusters_per_class=1)

def sigmoid(X):
    """
    Accepts X as a numpy array and applies a sigmoid transformation.
    
    sigmoid(x) = 1 / (1+ exp(-x))
    
    :returns: sigmoid(X)
    """
    
    denom = 1 + np.exp(-1 * X)
    sigmoid_X = 1 / denom
    
    return sigmoid_X

def accuracy_score(y, y_pred):
    """
    Compares prediction vector (y_pred) with targets (y) and returns the accuracy.
    
    accuracy = (True Pos. + True Neg.) / N

    :returns: Accuracy
    """
    # Initialize values
    True_Pos = 0
    True_Neg = 0
    
    # Compare values
    for i in range(len(y)):
        if y[i] == y_pred[i]:
            if y[i]:
                True_Pos += 1
            else:
                True_Neg += 1
    
    return (True_Pos + True_Neg) / len(y)
        

# Create linear space of X
linspace_X = np.linspace(-5, 5, 200)

# Test to see that sigmoid function is working
plt.plot(linspace_X, sigmoid(linspace_X))
plt.show()

# By taking a sigmoid of the data, we can already do some basic classification of the data
plt.scatter(X, [1 if i > 0.5 else 0 for i in sigmoid(X)], label="Classification")
plt.scatter(X, sigmoid(X), label = "Probability")
plt.legend()
plt.title("Classification of Points via Naive Logistic Regression (No Weights)")
plt.xlabel("Accuracy Score: {}".format(accuracy_score(y, [1 if i > 0.5 else 0 for i in sigmoid(X)])))
plt.show()