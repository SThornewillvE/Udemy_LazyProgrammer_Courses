# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 16:52:19 2019

@author: sthornewillvonessen
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve

plt.style.use("seaborn")


def precision_and_recall(y, y_pred):
    
    tp = 0
    fp = 0
    fn = 0
    
    for i in range(len(y)):
        if y[i] == y_pred[i] and y[i] == 1:
            tp+=1
        elif y[i] == y_pred[i] and y[i] == 0:    
            pass
        elif y[i] > y_pred[i]:
            fn+=1
        else:
            fp+=1
    
    try: 
        precision = tp/(tp+fp)
    except ZeroDivisionError: precision = 0
    
    try: 
        recall = tp/(tp+fn)
    except ZeroDivisionError: recall = 0
    
    return precision, recall


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

precision_over_threshold = []
recall_over_threshold = []

for threshold in np.linspace(0, 1, 200)[1:200]:

        y_pred2 = np.array([int(i) for i in p_pred[:, 0] < threshold])
        
        prec, rec = precision_and_recall(y_pred, y_pred2)
        
        precision_over_threshold.append(prec)
        recall_over_threshold.append(rec)


plt.plot(ones, ones)
plt.plot(precision_over_threshold, recall_over_threshold)
plt.show()