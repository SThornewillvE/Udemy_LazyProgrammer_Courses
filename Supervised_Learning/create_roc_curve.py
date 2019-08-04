# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 16:52:19 2019

@author: sthornewillvonessen
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, roc_auc_score

plt.style.use("seaborn")


def precision_and_recall(y, y_thresh):

    # Get confusion Matrix    
    cm = confusion_matrix(y, y_thresh).T
    
    # From confusion matrix, get sensitivity and specificity
    
    tp_rate = cm[0, 0]/(cm[0, 0] + cm[1, 0])  # Same as Precision and/or Sensitivity
    fp_rate = cm[0, 1]/(cm[0, 1] + cm[1, 1])  # Same as 1 - Specificity/Recall
    
    return tp_rate, fp_rate


# Get straight line for ROC
ones = [0, 1]    

# Create data using package
X, y = make_classification(n_samples=100, n_features=2,
                           n_informative=2, n_redundant=0, 
                           n_classes=2, n_clusters_per_class=1)


# Create and fit classifier
clf = GaussianNB()
clf.fit(X, y)

# Get probabilities
y_pred = clf.predict(X)
p_pred = clf.predict_proba(X)[:, 0]

# Initialize values
x_vec = []
y_vec = []

# Look over thresholds to find true positive and false positive rates
for threshold in np.linspace(0, 1, 200):

        y_thresh = np.array([int(i) for i in p_pred < threshold])
        
        tp_rate, fp_rate = precision_and_recall(y, y_thresh)

        # Save Values        
        x_vec.append(fp_rate)
        y_vec.append(tp_rate)

# Convert to np array
x_vec = np.array(x_vec)
y_vec = np.array(y_vec)

# Get AUC
auc = 1 - roc_auc_score(y, p_pred)

# Plot
plt.plot(ones, ones, label = "w/ all scores 0.5")
plt.plot(x_vec, y_vec, label = "ROC for model scores")
plt.title("ROC curve for predicted probabilities from model \n AUC = {}".format(round(auc, 3)))
plt.xlabel("False Positive Rate (1 - specificity)")
plt.ylabel("True Positive Rate (sensitvity)")
plt.legend()
plt.show()
