#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 21:36:48 2019

@author: simonthornewillvonessen
"""

# Import Packages
import numpy as np

def L1_Norm(X, y, Lambda = 1, alpha=0.0001, steps=1000):
    """
    Calculates L1_Norm to find the best weights to match X to targets(y).
    
    Note that Lambda is a weight set for how powerful this normalisation is and
    can be set as necessary.
    
    alpha and steps are hyperparameters used for gradient descent.
    
    :returns: Theta
    """
    
    # First we'll calculate n
    n = len(y)
    
    # Check that x_i and y_i are the same length
    if n != len(X):
        print("Error, X and y do not contain an equal number of observations")
        return
    
    # Initialize array for Theta
    Theta = np.array([100, 100, 100])
    
    for i in range(steps):

        # Find the gradient        
        Xt_X = np.matmul(X.T, X)        
        dJ_dTheta = (-2*np.matmul(X.T, y)) + (2*np.matmul(Xt_X, Theta)) + Lambda*np.sign(Theta)
        
        # Update weights
        Theta = Theta - (alpha * dJ_dTheta)  # X.T * loss is the gradient
    
    
def Main():
    pass


if __name__ == '__main__':
    Main()