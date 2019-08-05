# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 13:09:36 2019

@author: sthornewillvonessen
"""

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


plt.style.use("seaborn")
color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def resample(X, y):
    """
    Gets bootstrap sample of X and y
    """
    
    idx = [i for i in range(len(X))]  # Get indexes for random sampling
    
    sample = np.random.choice(idx, size=len(X), replace=True)

    return X[sample], y[sample]
    
# Generate random data for use
X, y = make_regression(n_samples=1000, n_features=1, noise = 100)


B = 1_000  # Number of models

# Create ensemble
models = []

for b in range(B):
    
    reg = LinearRegression()
    
    X_b, y_b = resample(X, y)
    
    reg.fit(X_b, y_b)
    
    models.append(reg)
    
# Create predictions
X_linsp = np.linspace(-3, 3, 1000).reshape(-1, 1)

preds = [m.predict(X_linsp) for m in models]
y_pred = np.empty(len(preds))

# Sum and normalise
for pred in preds:
    y_pred = y_pred + pred
y_pred = y_pred/len(y_pred)

# Plot Results
plt.plot(X_linsp, y_pred, label="bagged")
for mdl in models:
    plt.plot(X_linsp, mdl.predict(X_linsp), color=color_cycle[2], alpha=0.005)
plt.scatter(X, y, color=color_cycle[1], alpha=0.6)
plt.legend()
plt.show()