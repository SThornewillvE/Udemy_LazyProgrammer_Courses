# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 10:49:43 2019

@author: sthornewillvonessen
"""

# Import Packages
import numpy as np

def R_sq_eval(y, y_pred):
    """
    Compares predicted independant values and compares it with the real value to compare the R-squared value.
    
    :returns: r_sq
    """
    
    # Convert to array
    y = np.array(y)
    y_pred = np.array(y_pred)
    
    if len(y) != len(y_pred):
        print("Input y and y_pred are not the same length")
        return None
    

    d1 = y - y_pred
    d2 = y - y.mean()
    
    r2 = 1 - (d1.dot(d1) / d2.dot(d2))
    
    return r2
    

def Main():
    pass


if __name__ == '__main__':
    Main()