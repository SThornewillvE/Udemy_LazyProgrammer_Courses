# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 16:40:46 2019

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
# Define Forward Algorithm
# ======================================================================================================================

class forward:
    
    def __init__(self, A, B, pi):
        
        self.A = A
        self.B = B
        self.pi = pi
        
        self.M = A.shape[0]
        self.V = B.shape[1]
        
            
    def get_px(self, x):
        """
        Gets p(x) for a given sequence and model.
        
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
        
        alphas = np.zeros((T, self.M))
        
        # Get the first alpha
        alphas[0] = self.pi * self.B[:, x[0]]
        
        # Get the rest of the alphas
        for t in range(1, T):
            
            alpha = np.zeros((1, 2))
            b_ik = B[:, x[t]]
            
            for i in range(M):
                alpha += A[i] * alphas[t-1]
            
            alphas[t] = alpha * b_ik
               
            
        return alphas[-1].sum()


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
fw = forward(A, B, pi)

# Get probability of x
p_x = fw.get_px(x)

print("Probability of seeing {}th row of X\n({}):\n{}".format(i, x, p_x))
