# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 17:56:23 2019

@author: sthornewillvonessen
"""

import numpy as np

def L2_norm(X, y, Lambda=1000):
    """
    Performs a linear regression on the X given target data y and will return a set of weights Theta.
    
    Note that this regularisation will be done to the strength of Lambda, which can be set freely.
    
    :returns: Theta
    """
    
    # First we'll calculate n
    n = len(y)
    
    # Check that x_i and y_i are the same length
    try: 
        n == len(X)
    except:
        print("Error, X and y do not contain an equal number of observations")
        return
    
    Lambda_I = np.identity(X.shape[1]) * Lambda
    
    Lambda_I_p_Xt_X = Lambda_I + np.matmul(X.T, X)
    Xt_y = np.matmul(X.T, y)
    
    Theta = np.linalg.solve(Lambda_I_p_Xt_X, Xt_y)

    return Theta
    
    
def Main():
    pass


if __name__ == '__main__':
    Main()
