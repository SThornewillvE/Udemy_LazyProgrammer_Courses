# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 08:46:41 2019

@author: SimonThornewill
"""

# ======================================================================================================================
# Import Packages
# ======================================================================================================================

import numpy as np
import matplotlib.pyplot as plt

# ======================================================================================================================
# Define functions
# ======================================================================================================================

def simple_init(M, K, D):
    
    M, K, D = (1, 1, 1)
    pi = np.array([1])
    A = np.array([[1]])
    R = np.array([[1]])   
    mu = np.array([[[0]]])
    sigma = np.array([[[[1]]]])
    
    return M, K, D, pi, A, R, mu, sigma


def big_init(M, K, D):
    
    # Generate pi
    pi = np.zeros([1, M])
    pi[0, 0] = 1
    
    # Generate A
    A = np.zeros((M, M))
    
    for i in range(M):
        A[i, :] = (1-0.9)/(M-1)
        A[i, i] = 0.9
    
    # Generate R    
    R = np.ones((M, K)) / K
    
    # Generate mu
    mu_array = []
    for m in range(M):
        m_array = []
        mu_array.append(m_array)
        for k in range(K):
            m_array.append([k+ (M*m) for d in range(D)])
    
    mu = np.array(mu_array)
        
    # Generate sigma
    sigma = np.zeros((M, K, D, D))
    for m in range(M):
        for k in range(K):
            sigma[m, k] = np.eye(D)  # Note that eye creates the identity matrix of dimensions DxD
    
    return M, K, D, pi, A, R, mu, sigma
        

def get_signals(M, K, D, N=20, T=100, init=big_init):
    
    M, K, D, pi, A, R, mu, sigma = init(M, K, D)
    
    X = []
    
    for n in range(N):
        x = np.zeros((T, D))
        s = 0
        r = np.random.choice(K, p=R[s])
        x[0] = np.random.multivariate_normal(mu[s][r], sigma[s][r])
        
        for t in range(1, T):
            s = np.random.choice(M, p=A[s])
            r = np.random.choice(K, p=R[s])
            x[t] = np.random.multivariate_normal(mu[s][r], sigma[s][r])
        
        X.append(x)
        
    return X
    
    
# ======================================================================================================================
# Main
# ======================================================================================================================

if __name__ == '__main__':
    x_obs = get_signals(M=5, K=3, D=1, N=1, T=100, init=simple_init)
    
    plt.plot(range(100), list(x_obs[0][:, 0]))
    plt.show()
    
    plt.hist(list(x_obs[0][:, 0]))
    plt.show()