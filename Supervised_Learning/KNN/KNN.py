# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 10:21:39 2019

@author: sthornewillvonessen
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

class KNN():
    def __init__(self, k):
        self.k = k
        
        return
        
        
    def fit(self, X, y):
        """
        Fits data based on input X and y by saving the datasets.
        """
        self.X = X
        self.y = y
        
        return
    
    
    def predict(self, X_new):
        """
        Predicts datapoints in X based on fitted data.
        """
        
        # Get length of X
        X_n = len(X)
        
        # Reshape new value appropriately
        self.X_new = X_new.reshape(1, 2).repeat(X_n, axis=0)
    
        # Find relevant vectors
        diff = self.X -self.X_new
        
        # Calculate distances
        D = (diff ** 2)
        norms = np.sqrt(D.sum(1))
    
        # Find closest k points to new value
        i =  np.argpartition(norms, self.k)[:self.k]
        
        # Make and return prediction
        y_pred = np.argmax(np.bincount(np.array([y[j] for j in i])))
        
        # Convert to array
        y_pred = np.array(y_pred)
        
        return y_pred

# =============================================================================
# Create data for demonstration
# =============================================================================

# Create data using package
X, y = make_classification(n_samples=100, n_features=2, 
                           
                           n_informative=2, n_redundant=0, 
                           n_classes=2, n_clusters_per_class=1)

# Create KNN object
nearest = KNN(k=3)

# Train model
nearest.fit(X=X, y=y)

# Create test point 
X_new = np.array([0.5, 1])
y_pred = nearest.predict(X_new)

# Reshape Data for Plotting
X_plot = np.concatenate((X, X_new.reshape(1, 2)))
y_plot = np.concatenate((y.reshape(100, 1), y_pred.reshape(1, 1)))

for i in [0, 1]:
    plt.scatter(X[:, 0][y==i], X[:, 1][y==i],
                label="classification = {}".format(i))
plt.scatter(X_new[0], X_new[1], marker='x', label="classification = {}".format(y_pred))
plt.legend()
plt.show()