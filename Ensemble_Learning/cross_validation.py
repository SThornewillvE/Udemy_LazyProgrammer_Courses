# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 20:32:00 2019

@author: SimonThornewill
"""

# Import Libaries
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier as knn

plt.style.use("seaborn")


def train_test_split(X, y, prop_train=0.8):
    """
    Performs a train test split
    
    :Inputs:
        :X: Independent variables
        :y: Dependent variable
        :prop_train: Proportion of the dataset to be in the training set
    :returns: X_train, X_test, y_train, y_test
    """
    
    # Create a vector with indeces of X and y
    idx = [i for i in range(0, len(X))]
    
    # Choose train set at random
    idx_train = np.random.choice(idx, 
                                 size=int(len(X)*prop_train), 
                                 replace=False)
    
    idx_test = list(set(idx).difference(set(idx_train)))
    
    # Use indeces to get training and test sets
    X_train = X[idx_train]
    X_test = X[idx_test]

    y_train = y[idx_train]
    y_test = y[idx_test]
    
    return X_train, X_test, y_train, y_test


def train_cross_validation(model, X_train, y_train, k_folds=10):
    """
    Trains using K-fold cross validation and returns a list of errors.
    
    First it splits the model into k equal folds randomly
    
    Then a score is created by itrating over the folds and leaving one out each one.
    
    :Inputs:
        :model: Model to be evaluated
        :X_train: Training independent variables
        :y_train: Training labels
        :k_fold: Number of folds in cross validation
    """
    
    # Initialise list to keep scores
    scores = []
    
    # Split data into k parts
    idx = [i for i in range(len(X_train))]
    
    # Initialize values
    folds = []
    new_fold = []
    datum_number = 1
    
    # Draw i from shuffled idx
    for i in np.random.choice(idx, size=len(idx)):
        
        # Check if it is time for a new fold
        if not datum_number%(len(idx)/k_folds):
            new_fold.append(i)
            
            # Save and reset before next continuation
            folds.append(new_fold)
            new_fold = []
            
            datum_number+= 1
            
            continue
        
        else:
            new_fold.append(i)
            datum_number+= 1
        
    # train and test on folds and keep score
    for i in range(len(folds)):
        
        # Get train and test
        X_cv_test = X[folds[i]]
        y_cv_test = y[folds[i]]
        
        # Get the rest of the lists and concatenate them together. Yes, this is messy and I am sorry.
        X_cv_train = X[[inner for outer in [j for j in folds if j != folds[i]] for inner in outer]]
        y_cv_train = y[[inner for outer in [j for j in folds if j != folds[i]] for inner in outer]]
        
        # Fit Model
        model.fit(X_cv_train, y_cv_train)
        
        # Predict on data
        y_pred = model.predict(X_cv_test)
        
        # Append score
        scores.append(accuracy_score(y_cv_test, y_pred))
    
    return np.array(scores)


# Create data using package
X, y = make_classification(n_samples=1_000, n_features=10,
                           n_informative=2, n_redundant=8, 
                           n_classes=2, n_clusters_per_class=1)

# Split X and y into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, prop_train=0.9)

mean_scores = []
test_score = []

# Create and fit classifier
for i in range(100):
    clf = knn(n_neighbors=i+1)

    scores = train_cross_validation(clf, X_train, y_train)
    
    mean_scores.append(np.mean(scores))
    
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    test_score.append(accuracy_score(y_test, y_pred))
    
# Plot results
plt.plot(range(len(mean_scores)), mean_scores)
plt.plot(range(len(test_score)), test_score)
plt.show()

# Plot results
plt.plot(range(len(mean_scores)), np.array(mean_scores) - np.array(test_score))
plt.show()