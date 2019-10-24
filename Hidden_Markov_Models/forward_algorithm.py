# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 16:40:46 2019

@author: sthornewillvonessen

Note: This algorithm is implemented in exponential time, this is why it was not implemented as such in the lectures.
      As such, try not to use times above about 15
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
    
    def alpha(self, t, i):
        """
        Find the probability of a certain sequence given that it ended with an observation coming from a certain hidden 
        state.
        
        :Inputs:
            :t: Int. length of sequence
            :x: Int. Final observation
            :A: Array, MXM. Probability of going from i, j hidden state
            :B: Array, MxV. Probability of seeing observation j given hidden state j
            :pi: Array 1XM. Probability of starting in a certain hidden state
            :M: Int. Number of unique hidden states
        """
        
        print("t, i: ({}, {})".format(t, i))
        
        # Check for base case
        if t == 1:
            return self.pi[i] * self.B[i, self.x[t]]
        
        # Otherwise continue as normal
        else:
            sum_alphas = 0
            
            for i in range(M):
                for j in range(M):
                    sum_alphas += A[i, j] * self.alpha(t-1, i)
            
            return sum_alphas * self.B[self.x[t], i]
        
            
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
        
        self.x = x
        
        T = len(x)
        
        # Get probability of x
        p_x = 0
        
        for i in range(M):
            p_x += self.alpha(T-1, i)
            
        return p_x


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
