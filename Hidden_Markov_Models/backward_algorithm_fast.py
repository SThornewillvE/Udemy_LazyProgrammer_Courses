# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 10:25:34 2019

@author: sthornewillvonessen
"""

# ======================================================================================================================
# Import Functions
# ======================================================================================================================

import numpy as np
import pickle as p

# Define Seed
np.random.seed(42)


# ======================================================================================================================
# Define Backward Algorithm
# ======================================================================================================================

class backward:
    
    def __init__(self, A, B, pi):
        
        self.A = A
        self.B = B
        self.pi = pi
        
        self.M = A.shape[0]
        self.V = B.shape[1]
        
            
    def get_px(self, x):
        """
        Gets p(x) for the rest of a sequence at time t.
        
        :Inputs:
            :x: Array, 1xN. Observations
            :A: Array, MXM. Probability of going from i, j hidden state
            :B: Array, MxV. Probability of seeing observation j given hidden state j
            :pi: Array 1XM. Probability of starting in a certain hidden state
            :M: Int. Number of unique hidden states
        :Returns:
            :p_x: Float, probability of seeing given sequence
        """
        
        T = len(x)
        
        betas = np.zeros((T, self.M))
        
        betas[-1] = 1
        
        for t in range(T-2, -1, -1):
            betas[t] = A.dot((B[:, x[t+1]] * betas[t+1, :]))
        
        return betas


# ======================================================================================================================
# Main
# ======================================================================================================================

# Load X
with open("./X.pkl", "rb") as f:
    X = p.load(f)

# Get random observation from X
i = np.random.randint(len(X))
x = X[i][:3]

# Define Markov Model
A = np.array([[0.1, 0.9], [0.8, 0.2]])
B = np.array([[0.6, 0.4], [0.3, 0.7]])
pi = np.array([0.5, 0.5])

# From Markov Model, get other parameters
M = A.shape[0]

# Initialize forward algo.
bw = backward(A, B, pi)

# Get probability of x
betas = bw.get_px(x)

print("Betas for sequence {}:\n".format(x), betas)