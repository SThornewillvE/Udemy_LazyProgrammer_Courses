# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:57:21 2019

@author: sthornewillvonessen
"""

# Import packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

plt.style.use("seaborn")

# Import data
df = pd.read_csv("dat/advertisement_clicks.csv")

# Define functions
def convert_to_matrix(df):
    """
    Converts dataframe of two columns, one showing whether an observation is the control or the treatment group and 
    another showing whether this observation converted with 1s or 0s.
    
    :reutrns: 2x2 np_array
    """
    
    a = df[df["advertisement_id"] == "A"]["action"].sum()
    b = df[df["advertisement_id"] == "A"]["action"].count() - a
    c = df[df["advertisement_id"] == "B"]["action"].sum()
    d = df[df["advertisement_id"] == "B"]["action"].count() - c
    
    return np.matrix('{} {}; {} {}'.format(a, b, c, d))
    

def chi_squared(M):
    """
    Calculate approximation for Chi-Squared test value from 2x2 numpy matrix
    
    :returns: chi_sq
    """

    # Calculate total
    n = M[0].sum() + M[1].sum()
    a = M[0].item(0)
    b = M[0].item(1)
    c = M[1].item(0)
    d = M[1].item(1)

    # Calculate chi-sq
    chi_sq = (n*((b*c - a*d)**2))/((a + b)*(a + c)*(b + d)*(c + d))

    return chi_sq


def p_value(chi_sq):
    """
    Calculates p-value for chi-squared test given a test value
    
    :returns: p_value
    """
    
    return 1 - stats.chi2.cdf(chi_sq, 1)
    
            
# Convert Data to Matrix form            
M = convert_to_matrix(df)

# Find Chi-Sq. Value
chi_sq = chi_squared(M)

# Find p-value
p_val = p_value(chi_sq)  # For this data the value is > 0.05 and so the null hypothesis can be rejected