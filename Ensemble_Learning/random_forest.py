# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 15:09:24 2019

@author: sthornewillvonessen
"""

# =============================================================================
# Import packages
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import make_classification

plt.style.use("seaborn")


# =============================================================================
# Write class for classification
# =============================================================================

def entropy(y):
    """
    Calculate entropy for vector y.
    
    N.B. It is assumed that y is a binary categorical variable.
    
    :Inputs:
        :y: Vector of values
    :Returns:
        :e: The amount of disorder in y (entropy)
    """
    
    # assume y is binary - 0 or 1
    n = len(y)
    
    # Get the number of labels of 1
    s1 = (y == 1).sum()
    
    # If all values in a vector are the same then entropy is 0
    if s1==0 or n == s1:
        return 0
    
    # Get probability of drawing a 1 or 0
    p1 = float(s1) / n
    p0 = 1 - p1
    
    # Calculate entropy
    e = -p0*np.log2(p0) - p1*np.log2(p1)
    
    return e


class TreeNode_RF:
    """
    Creates a tree node that is split on the column giving the most information gain. (Each node will only have a max
    of two children.)
    
    Note this tree class has been adapted to suit random forest.
    
    :Attributes:
        :depth: Depth of node
        :max_depth: Max. permissible depth of node
        :col: Column with highest ig that is split on
        :split: Threshold used for the split
        :left: Child left node
        :right: Child right node
        :prediction: List, predicted values for if beneath or above threshold
    """

    def __init__(self, depth=0, max_depth=None, cols_to_sample=1):
        self.depth = depth
        self.max_depth = max_depth
        
        # Attributes that cannot be set by constructor
        self.col = None
        self.split = None
        self.left = None
        self.right = None
        self.D = cols_to_sample
      
        
    def fit(self, X, y):
        """
        Fits model to data X and y.
        
        Fitting is done by searching through each feature and creating a split based on which column gives the best
        information gain (ig).
        """
        
        # In the case of there being only one observation in the sample then fit node to this sample
        if len(y) == 1 or len(set(y)) == 1:
            self.prediction = y[0]
        
        # In the case that the length of vector y is larger than 1        
        else:
            
            # Create column indexes from X
            n_dims = X.shape[1]
            cols = range(n_dims)
            
            # RF Adaptation, select only random cols to fit on
            cols = np.random.choice(cols, self.D, replace=False)
            
            # Initialize values for finding optimum
            max_ig = 0
            best_col = None
            best_split = None
            
            for col in cols:
                
                # Find the best split in this column and the associated information gain (ig)
                ig, split = self.find_split(X, y, col)
                
                # If information gain is best yet, remember it
                if ig > max_ig:
                    max_ig = ig
                    best_col = col
                    best_split = split
            
            # In the case that no column gived better ig
            if max_ig == 0:
                self.col = None
                self.split = None
                self.left = None
                self.right = None
                self.prediction = np.round(y.mean())  # Can't just return y, since it is necessarily larger than 0
                
            else:
                self.col = best_col
                self.split = best_split
                
                # If the depth of this node is equal than the max depth then don't assign new left and right nodes
                if self.depth == self.max_depth:
                    self.left = None
                    self.right = None 
                    self.prediction = [np.round(y[X[:, best_col] < self.split].mean()),  # Prediction for beneath thresh
                                       np.round(y[X[:, best_col] >= self.split].mean())]  # Prediction for above thresh
                    
                else:  # Splits X and y for use in child nodes
                    
                    # Get indexes for left and right splits
                    left_idx = (X[:, best_col] < best_split)
                    right_idx = (X[:,best_col] >= best_split)
                    
                    # Create X and y for left and right child nodes
                    X_left = X[left_idx]
                    y_left = y[left_idx]
                    
                    X_right = X[right_idx]
                    y_right = y[right_idx]
                    
                    # Create child nodes, THROUGH EPIC RECURSION FOR THE FIRST TIME IN MY LIFE
                    self.left = TreeNode_RF(self.depth + 1, self.max_depth)
                    self.left.fit(X_left, y_left)
                    
                    self.right = TreeNode_RF(self.depth + 1, self.max_depth)
                    self.right.fit(X_right, y_right)
          
            
    def find_split(self, X, y, col):
        """
        Find split which creates the most information gain for column in X.
        
        :Inputs:
            :X: Independent features, matrix
            :y: Dependent varaible, vector
            :column to search:
        :Returns:
            :max_ig: Max. information gain
            :best_split: Split which gives max_ig
        """
        
        # Get values in column defined by function
        x_vals = X[:, col]
        
        # Sort the values to find split
        idx_sorted = np.argsort(x_vals)
        
        # Update values in sorted order
        x_vals = x_vals[idx_sorted]
        y_vals = y[idx_sorted]
        
        
        # Get the boundaries for the classification
        boundaries = np.nonzero(y_vals[:-1] != y_vals[1:])[0]
        
        # Initialise values to find best split and max ig
        best_split = None
        max_ig = 0
        
        for b in boundaries:
            
            # Get the mid point between two values and information gain
            split = (x_vals[b] + x_vals[b+1]) / 2  
            ig = self.information_gain(x_vals, y_vals, split)
            
            if ig > max_ig:
                max_ig = ig
                best_split = split
        
        return max_ig, best_split
    
    
    def information_gain(self, x, y, split):
        """
        Gets information gain best on a split of the data
        
        :Inputs:
            :x: Vector, values of x to split on
            :y: Vector, labels for x
            :split: Place where x should be split to valuate ig in y
        :Returns:
            :ig: Information gain based on split
        """
            
        # Split y into left and right parts
        y_0 = y[x<split]
        y_1 = y[x>=split]
        
        # Get number of samples
        n = len(y)
        len_y0 = len(y_0)
        
        if len_y0 == 0 or len_y0 == n:
            return 0  # Return 0 in the case that no split was made
        
        # Get probability of classifying zero or 1
        p_0 = float(len(y_0)) / n
        p_1 = 1 - p_0
        
        # Calculate information gain
        ig = entropy(y) - p_0*entropy(y_0) - p_1*entropy(y_1) 
        
        return ig
    
    def predict_one(self, x):
        """
        predicts the value for one observation (x)
        
        :Inputs:
            :x: Observation to be predicted
        :Returns:
            :pred: Prediction for x
        """
        
        if self.col is not None and self.split is not None:
            
            # Get value of feature node is splitting on
            feature = x[self.col]
            
            # Find out if feature belongs to left or right split
            # Recurse through left/right nodes until at the bottom, when it is so then make prediction
            if feature < self.split:
                if self.left:
                    p = self.left.predict_one(x)
                else:
                    p = self.prediction[0]
            
            else:
                if self.right:
                    p = self.right.predict_one(x)
                else:
                    p = self.prediction[1]
        
        # In the case of no split, just return prediction
        else:
            p = self.prediction
        
        return p
    
    def predict(self, X):
        """
        Predicts values for a matrix (X)
        """
        
        # Find the number of observations in X
        n = len(X)
        
        # Initialize array of predictions
        p = np.zeros(n)
        
        for i in range(n):
            p[i] = self.predict_one(X[i])
        
        return p
    
class DecisionTree_RF:
    """
    Object that contains root for TreeNodes, Note that this Tree is a random forest adaptation.
    """
    
    def __init__(self, max_depth=None, cols_to_sample=1):
        
        self.max_depth = max_depth
        self.cols_to_sample = cols_to_sample
        
    def fit(self, X, y):
        
        self.root = TreeNode_RF(max_depth=self.max_depth, cols_to_sample=self.cols_to_sample)
        self.root.fit(X, y)
    
    
    def predict(self, X):
        
        return self.root.predict(X)
    

class RandomForest:
    """
    Bootstrapping wrapper for DecisionTree_RF.
    """
    
    def __init__(self, cols_to_sample=1):
        self.cols_to_sample = 1
        self.trees = []
        
    def fit(self, X, y, n_learners=10):
        """
        Fits random forest to data.
        """
        
        for i in range(n_learners):
            
            # Initialize learner
            learner = DecisionTree_RF(cols_to_sample=self.cols_to_sample)
            
            data_idx = np.random.choice(range(len(X)), len(X), replace=True)
            
            # Get sample of data for fitting
            X_b = X[data_idx]
            y_b = y[data_idx]
            
            # Fit learner
            learner.fit(X_b, y_b)
            
            self.trees.append(learner)
    
    
    def predict(self, X):
        """
        Predict values for X
        """

        # Create dictionary to keep track of predictions
        preds_dict = dict()

        for i, learner in enumerate(self.trees):
            preds_dict["tree_{}".format(i)] = learner.predict(X)

        # Convert prediction dictionary to something easy to work with    
        df_preds = pd.DataFrame(preds_dict)

        # Get predictions
        preds = df_preds.T.mean().round().values
            
        return preds
    
    
    def score(self, X, y):
        """
        Get score for RF
        """
        
        p = self.predict(X)
        
        return np.mean(p == y)
    
    
# =============================================================================
# Create data for demonstration
# =============================================================================

# Create data using package
X, y = make_classification(n_samples=100, n_features=6,
                           n_informative=2, n_redundant=6-2, 
                           n_classes=2, n_clusters_per_class=1)

# Create Tree Object
clf = RandomForest()

# Train model
clf.fit(X=X, y=y)

# Create test point 
X_new = X[np.random.choice(range(X.shape[0]))]
X_new = [list(X_new)]
y_pred = clf.predict(X_new)

y_pred