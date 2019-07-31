# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 15:38:54 2019

@author: sthornewillvonessen
"""

import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal as mvn

# Create Data
is_money = [1, 0, 1, 0, 1, 1, 0, 0, 1, 0]
is_free =  [1, 1, 0, 1, 1, 0, 0, 1, 0, 0]
is_pills = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
is_spam =  [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

df = pd.DataFrame(dict(is_money = is_money,
                       is_free = is_free,
                       is_pills = is_pills,
                       is_spam = is_spam))

class_list = df.is_spam.unique().tolist()

X = df.iloc[:, :3].values
y = df.iloc[:, 3].values

class gaussianNB():
    def __init__(self):
        self.gauss_dict = dict()
        self.priors = dict()
        self.classes = None
        
    def fit(self, X, y):
        
        self.classes = np.unique(y).tolist()
        
        for c in self.classes:  # Loops over distinct classes in y
        
            # Get vectors for classification
            X_c = X[y == c]
            
            Xc_mean = X_c.mean(axis=0)
            Xc_var = np.diag(np.cov(X_c.T)) + 1e-6  # Adding really small number so that no numbers are zero
 
            # Set gaussians and priors           
            self.gauss_dict[c] = (Xc_mean, Xc_var)
            self.priors[c] = len(y[y==c])/len(y)
            
    def predict(self, X):
        
        predictions = []
        max_post = -np.inf
        best_class = None
        
        for x in X:  # x is a row-vector of length 3,
            for c in self.classes:
                mu, var = self.gauss_dict[c]  # mu and var are also row vectors of 3
                post = mvn.logpdf(x, mean=mu, cov=var) + np.log(self.priors[c])

                if post > max_post:
                    max_post = post
                    best_class = c
                
            predictions.append(best_class)
        
        return predictions
    
# Create and train
clf = gaussianNB()

clf.fit(X, y)

# Predict a document that has all words (money, free, pills)
clf.predict([[1, 1, 1],
             [1, 1, 0],
             [1, 0, 1],
             [0, 1, 1],
             [0, 0, 1],
             [0, 1, 0],
             [1, 0, 0],
             [0, 0, 0]])