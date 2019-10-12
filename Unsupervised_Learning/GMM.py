# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 12:48:11 2019

@author: SimonThornewill
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal

# ======================================================================================================================
# Implement Gaussian Mixture Model Algorithm
# ======================================================================================================================

class GaussianMixtureModel:
    
    def __init__(self, k, max_iter=20, smoothing=1e-2):
        self.k = k
        self.max_iter = max_iter
        self.smoothing = smoothing
        
    def fit(self, X):
        
        # Get shape of X
        N, dims = X.shape
        
        # Set initial values
        means = np.zeros((self.k, dims))  # I.e. centers of clusters
        self.responsibilities = np.zeros((N, self.k))  # I.e. loss of each point to a certain cluster
        covars = np.zeros((self.k, dims, dims))  # I.e. the covar/var of each dim with each other dim and each cluster
        pi = np.ones(self.k) / self.k  # Mixing coeff, must add to one, prior is uniform dist.
        
        self.costs = np.zeros(self.max_iter)
        weighted_pdfs = np.zeros((N, self.k))
        
        for k in range(self.k):
            
            # Use some point as the center of the GMM
            means[k] = X[np.random.choice(N)]
            covars[k] = np.diag(np.ones(dims))
        
        for i in range(self.max_iter):  # For each round of optimisation
            for k in range(self.k):  # For each cluster
                for n in range(N):  # For each point
                    
                    # Calculate weighted PDF, prob of k given params
                    weighted_pdfs[n, k] = pi[k] * multivariate_normal.pdf(X[n], means[k], covars[k])  
            
            for k in range(self.k):
                for n in range(N):
                    
                    # Calculate self.responsibilities
                    self.responsibilities[n, k] = weighted_pdfs[n, k] / weighted_pdfs[n, :].sum()
            
            for k in range(self.k):
                
                # Get sum of self.responsibilities for all clusters
                Nk = self.responsibilities[:, k].sum()
                pi[k] = Nk/N
                means[k] =  self.responsibilities[:, k].dot(X)/ Nk
                
                # Hard part is updating covars, linalg is used in order to speed up calculations
                delta = X - means[k] # N x D
                Rdelta = np.expand_dims(self.responsibilities[:,k], -1) * delta # multiplies R[:,k] by each col. of delta - N x D
                covars[k] = Rdelta.T.dot(delta) / Nk + np.eye(dims)*self.smoothing # D x D
            
            self.costs[i] = np.log(weighted_pdfs.sum(axis=1)).sum()
            if i > 0:
                if np.abs(self.costs[i] - self.costs[i-1]) < 0.1:
                    break
   
    
    def plot(self, X):
        
        plt.plot(self.costs)
        plt.title("Costs")
        plt.show()
        
        # Get random colours just like in KNN file
        random_colors = np.random.random((self.k, 3))
        colors = self.responsibilities.dot(random_colors)
        
        plt.scatter(X[:, 0], X[:, 1], c=colors)
        plt.show()
            
        

# ======================================================================================================================
# Create data
# ======================================================================================================================

def create_data(dims=2, s=4):
    """
    Creates data for clustering algorithm.
    
    :Input:
        :dims: Int. Number of dimensional space that clusters occupy
        :s: Float. Size of space between clusters
    :Returns:
        :X: Matrix. Some dimensional matrix with random points in clusters some spacing apart
    """
    
    mu1 = np.array([0, 0])
    mu2 = np.array([s+3, s+3])
    mu3 = np.array([0-1, s+2])
    
    n = 2_000
    
    X = np.zeros((n, dims))
    
    X[:1_200, :] = np.random.randn(1_200, dims)*2 + mu1
    X[1_200:1_800, :] = np.random.randn(600, dims) + mu2
    X[1_800:, :] = np.random.randn(200, dims)*0.5 + mu3
    
    return X


# ======================================================================================================================
# Run Main
# ======================================================================================================================
    
X = create_data()

clust = GaussianMixtureModel(k=3)

clust.fit(X)

clust.plot(X)
