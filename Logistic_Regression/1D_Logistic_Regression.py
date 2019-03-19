# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 09:10:49 2019

@author: sthornewillvonessen
"""

# Import Packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# Import custom packages
from cross_entropy import cross_entropy_eval

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

def grad_desc(X, y, alpha=0.01, steps=10000, regularisation=None, lmbd=None):
    """
    Performs gradient descent on a randomly initialized set of variables to predict y based on X using logistic regression,
    where y is a binary output.
    
    Alpha is the learning rate and the steps is the number of iterations done for this algorithm
    
    Note that if regularisation is to be performed then lambda must also be defined.
    
    :returns: Theta
    """
    
    if not (regularisation and lmbd):
        print("Both lambda and regularisation type need to be defined.")
        return
    
    Theta = np.random.randint(0, 100, size=X.shape[1])
    
    for i in range(steps):
        if regularisation == 'L2':
            Theta = Theta - (alpha * (X.T.dot(y - sigmoid(X.dot(Theta)))) + lmbd*Theta) 
        elif regularisation == 'L1':
            Theta = Theta - (alpha * (X.T.dot(y - sigmoid(X.dot(Theta)))) + lmbd*np.sign(Theta)) 
        else:
            Theta = Theta - (alpha * X.T.dot(y - sigmoid(X.dot(Theta))))
        y_pred = sigmoid(X.dot(Theta))
        if i % 1000:
            print("Accuracy: {}".format(accuracy_score(y, y_pred)))
            
    return Theta

def add_bias(X):
    """
    Adds bias term to X as a column of 1s.
    
    :returns: X
    """    
    
    X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
    
    return X

# Create linear space of X
linspace_X = np.linspace(-5, 5, 200)

# Test to see that sigmoid function is working
plt.plot(linspace_X, sigmoid(linspace_X))
plt.show()

# Calculate random weights
Theta = np.random.randint(0, 100, size=X.shape[1])

# Calculate Naive y_pred
y_pred = sigmoid(X.dot(Theta))
classification = np.array([0 if i < 0.5 else 1 for i in y_pred])

# By taking a sigmoid of the data, we can already do some basic classification of the data
plt.scatter(X, y_pred, label = "Predicted")
plt.scatter(X, y, label="Reality")
plt.plot(linspace_X, sigmoid(linspace_X * Theta), label = "Sigmoid", alpha=0.7)
plt.legend()
plt.title("Classification of Points via Naive Logistic Regression (No Weights)")
plt.xlabel("Accuracy Score: {}".format(accuracy_score(y, [1 if i > 0.5 else 0 for i in sigmoid(X)])))
plt.show()

# Find loss from cross entropy
loss = cross_entropy_eval(y, y_pred)

# Add bias term to X
X = add_bias(X)  # it was found the bias term doesnt help with this dataset

# Perform Gradient Descent
Theta = grad_desc(X, y, alpha=0.0001)
y_pred = sigmoid(X.dot(Theta))

# PLot Fitted Curve
plt.scatter(X[:, 0], y_pred, label = "Predicted")
plt.scatter(X[:, 0], y, label="Reality")
plt.plot(linspace_X, sigmoid(linspace_X * Theta[0]), label = "Sigmoid", alpha=0.7)
plt.legend()
plt.title("Classification of Points via Naive Logistic Regression With Weights")
plt.xlabel("Accuracy Score: {}".format(accuracy_score(y, [1 if i > 0.5 else 0 for i in sigmoid(X)])))
plt.show()

