# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 10:28:37 2019

@author: sthornewillvonessen
"""

# Import Files
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm

plt.style.use("seaborn")
color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]


B = 200  # Number of bootstrap samples
N = 20  # Length of X
X = np.random.randn(N)  # Random vector to be estimated via bootstrap

print("sample mean of X:", X.mean())

# Create empty vector for individual estimates
individual_estimates = np.empty(B)

# Perform bootstrap
for b in range(B):
    sample = np.random.choice(X, size=N)
    individual_estimates[b] = sample.mean()

# Calculate stats from bootstrap
bmean = individual_estimates.mean()
bstd = individual_estimates.std()

lower = bmean + norm.ppf(0.025)*bstd
upper = bmean + norm.ppf(0.975)*bstd

lower2 = X.mean() + norm.ppf(0.025)*X.std()/np.sqrt(N)
upper2 = X.mean() + norm.ppf(0.975)*X.std()/np.sqrt(N)

print("bootstrap mean of X:", bmean)

# Plot results
plt.hist(individual_estimates, bins=20, alpha = 0.6)
plt.axvline(x=lower, label="bootstrap interval")
plt.axvline(x=upper)
plt.axvline(x=lower2, color=color_cycle[1], label = "sample interval")
plt.axvline(x=upper2, color=color_cycle[1])
plt.legend()
plt.plot()