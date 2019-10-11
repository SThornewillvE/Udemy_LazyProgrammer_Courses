# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 21:55:58 2019

@author: SimonThornewill
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram, linkage


# ======================================================================================================================
# Implement Clustering Algorithm
# ======================================================================================================================

class aggglomerative:
    
    def __init__(self, link="ward"):
        self.linkage = link
        
        
    def fit(self, X):
        
        self.Z = linkage(X, self.linkage)
        
        return
        
    def plot(self):
        
        ax, fig = plt.subplots(figsize=(10, 6))
        plt.title("Linkage: {}".format(self.linkage))
        dendrogram(self.Z)
        plt.show()

        return
     
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

clust = aggglomerative(link="ward")

clust.fit(X)

clust.plot()
