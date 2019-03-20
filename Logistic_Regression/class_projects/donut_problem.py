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
N = 1000

D = 2

# Radii of donuts
R_inner = 5
R_outer = 10


# Generate length component of polar coordinate
R1 = np.random.randn(int(N/2)) + R_inner
R2 = np.random.randn(int(N/2)) + R_outer

# Generate angle component of polar coords
theta = 2*np.pi*np.random.random(int(N/2))


# Assemble polar coordinates
X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T
X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T
X = np.concatenate([X_inner, X_outer])

# Assign classification of points
T = np.array([0]*int((N/2)) + [1]*int((N/2)))

# Plot data
plt.scatter(X[:, 0], X[:, 1], c=T)
plt.show()


# =============================================================================
# Create Logistic Classification Model
# =============================================================================

# Create bias term
ones = np.array([1]*N).T.reshape((1000, 1))

# Calculate lengths of each point
r = np.zeros((N, 1))
for i in range(N):
    r[i] = np.sqrt(X[i, :].dot(X[i, :]))

# Assemble dataset
Xb = np.concatenate((ones, r, X), axis=1)

# Initialize Random Weights
w = np.random.rand(D + 2)

# Find predicted values based on w
z = Xb.dot(w)
Y = sigmoid(z)

# Perform gradient descent
learning_rate = 0.0001  # Optimal values have already been calculated
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

"""
Postscript Comments:
    
    I'm actually relatively impressed that we could solve this problem simply by adding another feature such as the
    length of the raddius. I was expecting that we would have to change out loss function somehow to accomodate for 
    the non-linearity of this problem.
"""