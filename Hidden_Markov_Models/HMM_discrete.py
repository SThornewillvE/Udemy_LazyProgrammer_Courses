# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 14:45:54 2019

@author: sthornewillvonessen
"""

# ======================================================================================================================
# Import Packages
# ======================================================================================================================

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================================================================
# Define HMM
# ======================================================================================================================

class hidden_markov_model:
    
    def __init__(self, M):
        """
        Creates a hidden markov model object with M hidden states.
        
        :Inputs:
            :M: Total number of hidden states
        """
        
        self.M = M

    def fit(self, X, max_iter=30, rand_seed = False):
        """
        Fits HMM to X
        
        :Inputs:
            :X: Matrix. Design Matrix
            :max_iter: Int. Number of optimisation rounds to be done
            :rand_seed: Int. Random seed for reproducible results
        """
        
        if rand_seed:
            np.random.seed(rand_seed)
        
        # Define variables
        V = max(max(x) for x in X) + 1  # Define number of unique outputs, ass. each unique output is a categorical int
        N = len(X)  # Get number of sequences
        self.pi = np.ones(self.M) / self.M  # Likelihood of starting in any state
        self.A = self.random_normalised(self.M, self.M)  # Likelihood of going from any state to another state
        self.B = self.random_normalised(self.M, V)  # Likelihood of observing V given state M
        costs = []  # Create list to log costs
        
        for epoch in range(max_iter):
            
            # Print summary every 10 epochs
            if epoch % 10 == 0:
                print("epoch: {}".format(epoch))
                
            # Initialize list for alphas, betas and probabilities
            alphas = []
            betas = []
            P = np.zeros(N)
            
            for n in range(N):  # Begin loop through each observation
                
                # Initialize more variables for this loop
                x = X[n]
                T = len(x)
                alpha = np.zeros((T, self.M))
                beta = np.zeros((T, self.M))
        
                # Set first alpha, i.e. the probability of observing each state at time 0
                alpha[0] = self.pi * self.B[:, x[0]]  # Prob of being in some initial state and seeing that state
                
                # Set last beta, from definition
                beta[-1] = 1
                
                # Calculate the rest of the alphas
                for t in range(1, T):  # Skip t=0, start range at 1
                    
                    # Calculate and save alpha
                    alpha[t] = alpha[t-1].dot(self.A) * self.B[:, x[t]]  
                
                # Calculate the rest of the betas
                for t in range(T-2, -1, -1):
                    
                    # Calculate and save beta
                    beta[t] = self.A.dot(self.B[:, x[t+1]] * beta[t+1])
                
                # Save values
                P[n] = alpha[-1].sum()
                alphas.append(alpha)
                betas.append(beta)
                
            # Calculate cost
            cost = np.sum(np.log(P))
            costs.append(cost)
                
            # Reestimate pi and B using alpha and beta
            self.pi = np.sum(np.array([(alphas[n][0] * betas[n][0])/P[n] for n in range(N)])) / N
            
            den_1 = np.zeros((self.M, 1))
            den_2 = np.zeros((self.M, 1))
            
            a_num = 0
            b_num = 0
                
            for n in range(N):
                x = X[n]
                T = len(x)
                
                den_1 += (alphas[n][:-1] * betas[n][:-1]).sum(axis=0, keepdims=True).T / P[n]
                den_2 += (alphas[n] * betas[n]).sum(axis=0, keepdims=True).T / P[n]
                
                a_num_n  = np.zeros((self.M, self.M))
                b_num_n = np.zeros((self.M, V))
                
                for i in range(self.M):
                    for j in range(self.M):
                        for t in range(T-1):
                            a_num_n[i, j] += alphas[n][t, i] * self.A[i,j] * self.B[j, x[t+1]] * betas[n][t+1,j]

                for i in range(self.M):
                    for j in range(V):
                        for t in range(T):
                            if x[t] == j:
                                b_num_n[i,j] += alphas[n][t,i] * betas[n][t,i]
                                
                
                a_num += a_num_n / P[n]
                b_num += b_num_n / P[n]
            
            # Set new A and B
            self.A = a_num / den_1
            self.B = b_num / den_2
            
        print("A:", self.A)
        print("B:", self.B)
        print("pi:", self.pi)
            
        
        # Show plot of costs
        plt.plot(costs)
        plt.show()
            
            
    def likelihood(self, x):
        
        T  = len(x)
        
        alpha = np.zeros((T, self.M))
        alpha[0] = self.pi * self.B[:, x[0]]
        
        for t in range(1, T):
            alpha[t] = alpha[t-1].dot(self.A) * self.B[:, x[t]]
            
        return alpha[-1].sum()
    
    
    def likelihood_multi(self, X):
        return np.array([self.likelihood(x) for x in X])
    
    
    def log_likelihood_multi(self, X):
        return np.log(self.likelihood_multi(X))
    
    
    def get_state_sequence(self, x):
        T = len(x)
        delta = np.zeros((T, self.M))
        psi = np.zeros((T, self.M))
        
        delta[0] = self.pi * self.B[:, x[0]]
        for t in range(1, T):
            for j in range(self.M):
                delta[t,j] = np.max(delta[t-1]*self.A[:,j]) * self.B[j, x[t]]
                psi[t,j] = np.argmax(delta[t-1]*self.A[:,j])
                
        states = np.zeros(T, dtype=np.int32)
        states[T-1] = np.argmax(delta[T-1])
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        
        return states
    
    
    def random_normalised(self, M, N):
        """
        Returns a matrix with dims (M X N), this matrix has rows that sums to 1
        
        :Inputs:
            :M: Int. Rows in matrix
            :N: Int. Columns in matrix
        :Returns:
            :X: Matrix. As described above
        """
        
        X = np.random.random((M, N))
        
        return X / X.sum(axis=1, keepdims=True)
    
    
# ======================================================================================================================
# Main
# ======================================================================================================================
        
# Define X
X = []
for line in open('./dat/coin_data.txt'):
    # 1 for H, 0 for T
    x = [1 if e == 'H' else 0 for e in line.rstrip()]
    X.append(x)

# Create instance of HMM class    
hmm = hidden_markov_model(2)

# Fit hmm
hmm.fit(X, rand_seed = 42)
L = hmm.log_likelihood_multi(X).sum()
print("LL with fitted params:", L)

# try true values
hmm.pi = np.array([0.5, 0.5])
hmm.A = np.array([[0.1, 0.9], [0.8, 0.2]])
hmm.B = np.array([[0.6, 0.4], [0.3, 0.7]])
L = hmm.log_likelihood_multi(X).sum()
print("LL with true params:", L)

print("Best state sequence for:", X[0])
print(hmm.get_state_sequence(X[0]))