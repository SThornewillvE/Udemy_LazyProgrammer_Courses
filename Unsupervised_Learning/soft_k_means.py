# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 10:37:16 2019

@author: sthornewillvonessen
"""


# ======================================================================================================================
# Import Packages
# ======================================================================================================================

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================================================================
# Implement Clustering Algorithm
# ======================================================================================================================


class soft_kmeans:
    
    def __init__(self, k=3, max_iter=20, beta=1.0):
        self.k = k
        self.max_iter = max_iter
        self.beta = beta
        
        
    def fit(self, X):
        
        n, dims = X.shape
        
        self.centroids = np.zeros((self.k, dims))
        self.responsibility = np.zeros((n, self.k))
        costs = np.zeros(self.max_iter)
        
        # Initialise random points for centroids
        for i in range(self.k):
            self.centroids[i] = X[np.random.choice(n)]
        
        for i in range(self.max_iter):
            for j in range(self.k):
                for n in range(n):                    
                    self.responsibility[n, j] = np.exp(-self.beta*self.d(u=self.centroids[j], v=X[n])) /\
                    np.sum(np.fromiter((np.exp(-self.beta*self.d(u=self.centroids[l], v=X[n])) for l in range(self.k)), dtype=float))
            
            for j in range(self.k):
                self.centroids[j] = self.responsibility[:, j].dot(X) / self.responsibility[:, j].sum()
                
            costs[i] = self.cost(X, self.responsibility, self.centroids)
            if i > 0:
                if np.abs(costs[i] - costs[i-1] < 0.1):
                    break
    
    def plot_result(self, X):
        
        random_colors = np.random.random((self.k, 3))
        colors = self.responsibility.dot(random_colors)
        
        plt.scatter(X[:, 0], X[:, 1], c=colors)
        plt.show()
    
    def cost(self, X, r, cent):
        
        cost = 0
        
        for k in range(len(cent)):
            for n in range(len(X)):
                cost += r[n, k] * self.d(cent[k], X[n])
        
        return cost
        
    
    def d(self, u, v):
        diff = u - v
        return diff.dot(diff)
        
# ======================================================================================================================
# Create data
# ======================================================================================================================

def create_data(dims=2, s=4):
    mu1 = np.array([0, 0])
    mu2 = np.array([s, s])
    mu3 = np.array([0, s])
    
    n = 900
    
    X = np.zeros((n, dims))
    
    X[:300, :] = np.random.randn(300, dims) + mu1
    X[300:600, :] = np.random.randn(300, dims) + mu2
    X[600:, :] = np.random.randn(300, dims) + mu3
    
    return X


# ======================================================================================================================
# Run Main
# ======================================================================================================================
    
X = create_data()

# Plot Data
plt.scatter(X[:, 0], X[:, 1])
plt.plot()

clust = soft_kmeans()

clust.fit(X)

clust.plot_result(X)
