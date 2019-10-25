# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 14:46:29 2019

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

class Viterbi:
    

    def __init__(self, A, B, pi):
        
        self.A = A
        self.B = B
        self.pi = pi
        
        self.M = A.shape[0]
        self.V = B.shape[1]
    
    
    def most_likely_seq(self, x):
        """
        Generates the most likely sequence of observations given a markov model.
        
        :Inputs:
            :x: Array. List of random observations
        :Returns:
            :seq: Array. What was to be generated
        """
        
        T = len(x)
        
        deltas = np.zeros((T, self.M))
        psis = np.zeros((T, self.M))
        
        deltas[0] = self.pi * self.B[:, x[0]]
        for t in range(T):
            for j in range(self.M):
                
                deltas[t,j] = np.max(deltas[t-1]*self.A[:,j]) * self.B[j, x[t]]
                psis[t,j] = np.argmax(deltas[t-1]*self.A[:,j])
                
        seq = np.zeros(T, dtype=np.int32)
        seq[T-1] = np.argmax(deltas[T-1])
        for t in range(T-2, -1, -1):
            seq[t] = psis[t+1, seq[t+1]]
        
        return seq
        

# ======================================================================================================================
# Main
# ======================================================================================================================

# Load X
with open("./X.pkl", "rb") as f:
    X = p.load(f)

# Get random observation from X
i = np.random.randint(len(X))
x = X[i][:2]

# Define Markov Model
A = np.array([[0.1, 0.9], [0.8, 0.2]])
B = np.array([[0.6, 0.4], [0.3, 0.7]])
pi = np.array([0.5, 0.5])

# Create instance of viterbi
gen = Viterbi(A, B, pi)

# Get most likely sequence given a model
x_star = gen.most_likely_seq(x)

print(x_star)
