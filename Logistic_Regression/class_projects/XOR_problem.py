# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 11:01:49 2019

@author: sthornewillvonessen
"""

# =============================================================================
# Import packages
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn")


# =============================================================================
# Write Functions
# =============================================================================
def sigmoid(z):
    """
    Transforms vector z according to a sigma function.
    """
    return 1/(1+np.exp(-z))

def cross_entropy(T, Y):
    """
    Calculates cross entropy between target and predicted values.
    """
    E = 0
    for i in range(len(T)):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    
    return E

# =============================================================================
# Generate Data
# =============================================================================
# Total number of data points
N = 4

D = 2

X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    
T = np.array([0, 1, 1, 0])

plt.scatter(X[:, 0], X[:, 1], c=T)
plt.show()

# =============================================================================
# Create Logistic Classification Model
# =============================================================================

# Create bias term
ones = np.array([1]*N).T.reshape((4, 1))

# Calculate lengths of each point
xy = np.array([[1]*N]).T

# Assemble dataset
Xb = np.concatenate((ones, xy, X), axis=1)

# Initialize Random Weights
w = np.random.rand(D + 2)

# Find predicted values based on w
z = Xb.dot(w)
Y = sigmoid(z)

# Perform gradient descent
learning_rate = 0.001  # Optimal values have already been calculated
error = []
for i in range(5000):
    e = cross_entropy(T, Y)
    error.append(e)
    if i % 100 == 0:
        print("Erorr:", e)
        
    # Update w based on gradient descent
    w += learning_rate * (np.dot((T - Y).T, Xb) - 0.01*w)  # Lambda decided beforehand
    
    Y = sigmoid(Xb.dot(w))

# Plot error
plt.plot(error)
plt.title("Cross-entropy")
plt.show()

print("Final w:", w)  # Note that classification doesnt really depend on x, y coordinates. But it does depend on the radius
print("Final classification rate:", 1 - np.abs(T - np.round(Y)).sum() / N)
