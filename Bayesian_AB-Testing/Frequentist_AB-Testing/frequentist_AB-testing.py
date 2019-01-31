# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 12:53:43 2019

@author: sthornewillvonessen
"""

# Import Required Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats

# Use nice style for mpl
plt.style.use("seaborn")

# Write functions
def t_statistic(x1, x2, tails = 1):
    """
    x1: List of measurements for control
    x2: List of measurements for test
    
    :returns: t_stat
    """
    
    # Lind length of two vectors
    len_x1 = len(x1)
    len_x2 = len(x2)
    
    if len_x1 == len_x2:
        pooled_var = np.sqrt((x1.std() + x2.std()) * .5)
        t_stat = (x1.mean() - x2.mean())/(pooled_var * np.sqrt(2/(len_x1*2)))
    else:
        pooled_var = np.sqrt((len_x1*x1.std() + len_x2*x2.std()) / (len_x1 + len_x2 - 2))
        t_stat = (x1.mean() - x2.mean())/(pooled_var * np.sqrt(2/((1/len_x1) + (1/len_x2))))
        
    return t_stat


def p_value(t_stat, DoF, tails = 1):
    """
    Calculates the p-value based on the t-statistics
    
    :returns: p-value
    """
    
    return stats.t.cdf(t_stat, df=DoF) * tails


def degrees_of_freedom(x1, x2):
    """
    Calculates the degrees of freedom based on two inputs
    
    :returns: dof
    """
    
    if len(x1) == len(x2):
        dof = (2 * len(x1)) - 2
    else:
        dof = len(x1) + len(x2) - 2
    
    return dof


# Import Data
df = pd.read_csv("dat/advertisement_clicks.csv")

# Separate values into two
x_A = np.array(df[df["advertisement_id"] == 'A'].action)
x_B = np.array(df[df["advertisement_id"] == 'B'].action)

# Create histogram for run
plt.hist(x_A, bins = 2, alpha = .7, label='men')
plt.hist(x_B, bins = 2, alpha = .7, label='women')
plt.legend()
plt.show()

# Calculate t-stat
t_stat = t_statistic(x_A, x_B)

print("T-Statistic =", t_stat)

# Calculate p-value
p_val = p_value(t_stat, degrees_of_freedom(x_A, x_B),  tails = 2)

print("P-value =", p_val)

# Since the P-value is less than 5%, we can reject the null hypothesis