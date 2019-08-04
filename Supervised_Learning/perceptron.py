# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 16:09:00 2019

@author: SimonThornewill
"""

# =============================================================================
# Import packages
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification

plt.style.use("seaborn")

# =============================================================================
# Write class for classification
# =============================================================================

class Perceptron():
    
    def __init__(self, learning_rate=1.0, epochs=1_000):
        self.eta = learning_rate
        self.epochs = epochs
  
        # Get dimensions
        dims = X.shape[1]
        
        # Create random weights
        self.w = np.random.randn(dims)
        self.b = 0      
        
    def fit(self, X, y):
        
        # Initialise costs list and get length of y
        n = len(y)
        costs = []
        
        # Begin fitting process
        for epoch in range(self.epochs):
            
            # Make prediction
            y_pred = self.predict(X)
            
            # Find incorrect vals
            wrong = np.nonzero(y != y_pred)[0]
            
            if len(wrong) == 0:
                break  # Finish fitting if no incorrect
                
            # Select randomg wrong and update weights based on it
            i = np.random.choice(wrong)
            
            self.w += self.eta*y[i]*X[i]
            self.b += self.eta*y[i]
            
            # Get cost and remember it
            cost = len(wrong)/n
            costs.append(cost)
        
        print(" Final w: {}\n final b: {}".format(self.w, self.b))
        
    def predict(self, X):
        
        return np.sign(X.dot(self.w) + self.b)
    
    def score(self, X, y):
        
        P = self.predict(X)
        
        return np.mean(P == y)

# =============================================================================
# Create data for demonstration
# =============================================================================

# Create data using package
X, y = make_classification(n_samples=100, n_features=2, 
                           
                           n_informative=2, n_redundant=0, 
                           n_classes=2, n_clusters_per_class=1)

# Create Perceptron object
clf = Perceptron()

# Train model
clf.fit(X=X, y=y)

# Create test point 
X_new = np.array([0.5, 1])
y_pred = clf.predict(X_new)

# Reshape Data for Plotting
X_plot = np.concatenate((X, X_new.reshape(1, 2)))
y_plot = np.concatenate((y.reshape(100, 1), y_pred.reshape(1, 1)))

for i in [0, 1]:
    plt.scatter(X[:, 0][y==i], X[:, 1][y==i],
                label="classification = {}".format(i))
plt.scatter(X_new[0], X_new[1], marker='x', label="classification = {}".format(y_pred))
plt.legend()
plt.show()