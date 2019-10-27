# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 09:53:58 2019

@author: SimonThornewill
"""

# ======================================================================================================================
# Import Packages
# ======================================================================================================================

import numpy as np
import matplotlib.pyplot as plt

from generate_cont_dat_for_HMM import get_signals, big_init, simple_init
from scipy.stats import multivariate_normal as mvn

from tqdm import trange

# ======================================================================================================================
# Define HMM
# ======================================================================================================================

class hidden_markov_model_cont:
    
    def __init__(self, M, K):
        """
        Creates a hidden markov model object with M hidden states.
        
        :Inputs:
            :M: Total number of hidden states
            :K: Total number of Gaussians
        """
        
        self.M = M
        self.K = K
        

    def fit(self, X, max_iter=30, epsilon=10e-1, rand_seed = False):
        """
        Fits HMM to X
        
        :Inputs:
            :X: Matrix. Design Matrix
            :max_iter: Int. Number of optimisation rounds to be done
            :epsilon: Float. Variable used for smoothing
            :rand_seed: Int. Random seed for reproducible results
        """
        
        if rand_seed:
            np.random.seed(rand_seed)
        
        # train the HMM model using the Baum-Welch algorithm
        # a specific instance of the expectation-maximization algorithm

        N = len(X)
        D = X[0].shape[1] # assume each x is organized (T, D)

        self.pi = np.ones(self.M) / self.M # initial state distribution
        self.A = self.random_normalised(self.M, self.M) # state transition matrix
        self.R = np.ones((self.M, self.K)) / self.K # mixture proportions
        print("initial A:", self.A)
        print("initial R:", self.R)
        self.mu = np.zeros((self.M, self.K, D))
        for i in range(self.M):
            for k in range(self.K):
                random_idx = np.random.choice(N)
                x = X[random_idx]
                random_time_idx = np.random.choice(len(x))
                self.mu[i,k] = x[random_time_idx]
        self.sigma = np.zeros((self.M, self.K, D, D))
        for j in range(self.M):
            for k in range(self.K):
                self.sigma[j,k] = np.eye(D)

        self.costs = []
        for epoch in trange(max_iter):
            alphas = []
            betas = []
            gammas = []
            Bs = []
            # components = []
            P = np.zeros(N)

            for n in range(N):
                x = X[n]
                T = len(x)

                # calculate B so we can lookup when updating alpha and beta
                B = np.zeros((self.M, T))
                component = np.zeros((self.M, self.K, T)) # we'll need these later
                for j in range(self.M):
                    for t in range(T):
                        for k in range(self.K):
                            p = self.R[j,k] * mvn.pdf(x[t], self.mu[j,k], self.sigma[j,k])
                            component[j,k,t] = p
                            B[j,t] += p
                Bs.append(B)

                alpha = np.zeros((T, self.M))
                alpha[0] = self.pi*B[:,0]
                for t in range(1, T):
                    alpha[t] = alpha[t-1].dot(self.A) * B[:,t]
                P[n] = alpha[-1].sum()
                assert(P[n] <= 1)
                alphas.append(alpha)

                beta = np.zeros((T, self.M))
                beta[-1] = 1
                for t in range(T - 2, -1, -1):
                    beta[t] = self.A.dot(B[:,t+1] * beta[t+1])
                betas.append(beta)

                # update for Gaussians
                gamma = np.zeros((T, self.M, self.K))
                for t in range(T):
                    alphabeta = (alphas[n][t,:] * betas[n][t,:]).sum()
                    for j in range(self.M):
                        factor = alphas[n][t,j] * betas[n][t,j] / alphabeta
                        # mixture_j = component[j,:,t].sum()
                        for k in range(self.K):
                            gamma[t,j,k] = factor * component[j,k,t] / B[j,t]
                gammas.append(gamma)

            cost = np.log(P).sum()
            self.costs.append(cost)

            # now re-estimate pi, A, R, mu, sigma
            self.pi = np.array([(alphas[n][0] * betas[n][0])/P[n] for n in range(N)]).sum() / N

            a_den = np.zeros((self.M, 1))
            a_num = 0
            r_num = np.zeros((self.M, self.K))
            r_den = np.zeros(self.M)
            mu_num = np.zeros((self.M, self.K, D))
            sigma_num = np.zeros((self.M, self.K, D, D))
            for n in range(N):
                x = X[n]
                T = len(x)
                B = Bs[n]
                gamma = gammas[n]

                # denominator for A
                a_den += (alphas[n][:-1] * betas[n][:-1]).sum(axis=0, keepdims=True).T / P[n]

                # numerator for A
                a_num_n = np.zeros((self.M, self.M))
                for i in range(self.M):
                    for j in range(self.M):
                        for t in range(T-1):
                            a_num_n[i,j] += alphas[n][t,i] * self.A[i,j] * B[j,t+1] * betas[n][t+1,j]
                a_num += a_num_n / P[n]


                # update mixture components
                r_num_n = np.zeros((self.M, self.K))
                r_den_n = np.zeros(self.M)
                for j in range(self.M):
                    for k in range(self.K):
                        for t in range(T):
                            r_num_n[j,k] += gamma[t,j,k]
                            r_den_n[j] += gamma[t,j,k]
                r_num += r_num_n / P[n]
                r_den += r_den_n / P[n]

                mu_num_n = np.zeros((self.M, self.K, D))
                sigma_num_n = np.zeros((self.M, self.K, D, D))
                for j in range(self.M):
                    for k in range(self.K):
                        for t in range(T):
                            # update means
                            mu_num_n[j,k] += gamma[t,j,k] * x[t]

                            # update covariances
                            sigma_num_n[j,k] += gamma[t,j,k] * np.outer(x[t] - self.mu[j,k], x[t] - self.mu[j,k])
                mu_num += mu_num_n / P[n]
                sigma_num += sigma_num_n / P[n]

            self.A = a_num / a_den
            assert(np.all(self.A <= 1))

            # update R, mu, sigma
            for j in range(self.M):
                for k in range(self.K):
                    self.R[j,k] = r_num[j,k] / r_den[j]
                    self.mu[j,k] = mu_num[j,k] / r_num[j,k]
                    self.sigma[j,k] = sigma_num[j,k] / r_num[j,k]

        print("A:", self.A)
        print("mu:", self.mu)
        print("sigma:", self.sigma)
        print("R:", self.R)
        print("pi:", self.pi)
                
            
    def likelihood(self, x):
        T = len(x)
        alpha = np.zeros((T, self.M))

        B = np.zeros((self.M, T))
        for j in range(self.M):
            for t in range(T):
                for k in range(self.K):
                    p = self.R[j,k] * mvn.pdf(x[t], self.mu[j,k], self.sigma[j,k])
                    B[j,t] += p

        alpha[0] = self.pi*B[:,0]
        for t in range(1, T):
            alpha[t] = alpha[t-1].dot(self.A) * B[:,t]
        return alpha[-1].sum()
    
    
    def likelihood_multi(self, X):
        return np.array([self.likelihood(x) for x in X])
    
    
    def log_likelihood_multi(self, X):
        return np.log(self.likelihood_multi(X))
    
    
    def get_state_sequence(self, x):
        # returns the most likely state sequence given observed sequence x
        # using the Viterbi algorithm
        T = len(x)

        # make the emission matrix B
        B = np.zeros((self.M, T))
        for j in range(self.M):
            for t in range(T):
                for k in range(self.K):
                    p = self.R[j,k] * mvn.pdf(x[t], self.mu[j,k], self.sigma[j,k])
                    B[j,t] += p

        # perform Viterbi as usual
        delta = np.zeros((T, self.M))
        psi = np.zeros((T, self.M))
        delta[0] = self.pi*B[:,0]
        for t in range(1, T):
            for j in range(self.M):
                delta[t,j] = np.max(delta[t-1]*self.A[:,j]) * B[j,t]
                psi[t,j] = np.argmax(delta[t-1]*self.A[:,j])

        # backtrack
        states = np.zeros(T, dtype=np.int32)
        states[T-1] = np.argmax(delta[T-1])
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        return states
    
    def set_params(self, pi, A, R, mu, sigma):
        self.pi = pi
        self.A = A
        self.R = R
        self.mu = mu
        self.sigma = sigma
        M, K = R.shape
        self.M = M
        self.K = K
    
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
    
    def plot(self):
        plt.plot(self.costs)
        plt.show()
    
    
# ======================================================================================================================
# Main
# ======================================================================================================================
   
# Create Fake Signal (simple)
M, K, D= (1, 1, 1)
signals = get_signals(M, K, D, N = 10, T = 10, init=simple_init)

# Create instance of HMM class    
hmm = hidden_markov_model_cont(M=2, K=2)

# Fit hmm
hmm.fit(signals, rand_seed = 42)
L = hmm.log_likelihood_multi(signals).sum()
print("LL with fitted params:", L)

hmm.plot()
