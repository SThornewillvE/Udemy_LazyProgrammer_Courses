# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 09:42:28 2019

@author: sthornewillvonessen
"""


# Import Packages
import numpy as np

def cross_entropy_eval(y, y_pred):
    """
    Calculates cross entropy error for predictions (y_pred) and their targets (y)
    
    return: cross_entropy
    """
    
    cross_entropy = -1*(y*np.log(y_pred) + (1 - y)*np.log(1 - y_pred)).sum()
    
    return cross_entropy


def Main():
    pass


if __name__ == '__main__':
    Main()