# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 12:42:11 2019

@author: SimonThornewill
"""

# =============================================================================
# Import packages
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

plt.style.use("seaborn")


# =============================================================================
# Write class for classification
# =============================================================================

class Adaboost:
    
    def __init__(self, M):
        self.M = M
        
        # Initialize models and alphas lists
        self.models = []
        self.alphas = []
        
    def fit(self, X, y):
        
        # Get dimensions of matrix X
        n, d = X.shape
        
        # initialize weights
        theta = np.ones(n)/n
        
        for m in range(self.M):
            # Initialize Decision stump
            tree = DecisionTreeClassifier(max_depth=1)
            
            # Fit stump to data, with sample weights
            tree.fit(X, y, sample_weight=theta)
            
            # Get predictions
            y_pred = tree.predict(X)
            
            # Calculate error and alpha
            # Multiplies each incorrect classification by the weight of that obs and takes sum
            error = theta.dot(y_pred != y)  
            alpha = 0.5*(np.log(1-error) - np.log(error))
            
            # Recalculate theta based on loss and normalise
            theta = theta*np.exp(-alpha*y*y_pred)
            theta = theta/theta.sum()
            
            self.models.append(tree)
            self.alphas.append(alpha)
    
    
    def predict(self, X):
        """
        Note, not the same as SKlearn API
        """
        
        # Get dimensions of matrix X
        n, d = X.shape
        
        # Initialize list of predictions weighted by alpha
        FX = np.zeros(n)
        
        for alpha, tree in zip(self.alphas, self.models):
            FX += alpha*tree.predict(X)
            
        return np.sign(FX), FX
    
    
    def score(self, X):
        """
        Note, not the same as SKlearn API
        """
        
        y_pred, FX = self.predict(X)
        
        # Calculate loss
        L = np.exp(-y*FX).mean()
            
        return np.mean(y_pred == y), L
            

# =============================================================================
# Create data for demonstration
# =============================================================================

# Create data using package
X, y = make_classification(n_samples=100, n_features=2,
                           n_informative=2, n_redundant=0, 
                           n_classes=2, n_clusters_per_class=1)

# Correct y format
y = y*2 - 1

# Create Tree Object
clf = Adaboost(M=100)

# Train model
clf.fit(X=X, y=y)

# Create test point 
X_new = np.array(np.array([[0, -2]]))
y_sign, y_pred = clf.predict(X_new)

# Reshape Data for Plotting
X_plot = np.concatenate((X, X_new.reshape(1, 2)))
y_plot = np.concatenate((y.reshape(100, 1), y_pred.reshape(1, 1)))
