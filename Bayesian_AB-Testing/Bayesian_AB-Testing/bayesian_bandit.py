# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 15:07:56 2019

@author: sthornewillvonessen
"""

# Import packages
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.style.use('seaborn')

class Bandit:
    """
    Creates Bandit class with given probability and with default priors.
    """
    
    def __init__(self, p, a=1, b=1):
        self.p = p
        self.a = a
        self.b = b
        self.x = np.NaN
        
    def pull(self):
        """
        Returns bool value of whether a random number from uniform dist. between 0 and 1 is lower than the probability
        assigned to this bandit
        
        :returns: True/False (bool)
        """
        
        return np.random.random() < self.p
    
    def sample(self):
        """
        Samples from Beta distribution based on a and b. If a and b are unknown then a uniform distribution is assumed
        (a=1 and b=1)
        
        :returns: Number between 0 and 1
        """
        
        x = stats.beta.rvs(self.a, self.b, size=1)[0]
        
        return x
    
    def update(self, x):
        """
        Updates the parameters of the beta distribution based on what value was returned by sampling.
        
        Note that we use a beta distribution because it has values between 0 and 1 and with the parameters a and b you
        can describe a large variation of pdfs. 
        
        :returns:
        """
        self.a += x
        self.b += 1 - x
    
def plot(bandits):
    """
    Plots result of experiment after all of N has been performed using the bandits for the experiment.
    
    :returns:
    """
    x = np.linspace(0, 1, 200)
    for b in bandits:
        y = stats.beta.pdf(x, b.a, b.b)
        plt.plot(x, y)
    plt.show()
        

def run_experiment(p1, p2, N):
    """
    Run thompson sampling experiment with probabilities p1 and p2 and with a total sample size of N.
    
    :returns:
    """
    
    
    # Create list of bandits
    bandits = [Bandit(p) for p in [p1, p2]]
    
    # Loop over the number of rounds given by N
    for i in range(N):
        
        # Initialize important values
        bestb = None
        maxsample = -1
        allsamples = []
        
        # Get a sample for each bandit, update variables for best bandit
        for b in bandits:
            
            # Get sample from beta distribution associated with bandit
            sample = b.sample() 
            allsamples.append("%.4f" % sample)
            
            # Check if the sample for this bandit is the best, if so remember this bandit
            if sample > maxsample: 
                maxsample = sample
                bestb = b
        
        x = bestb.pull()  # Get True/False depending on if a random uniform number is greater than the probability of that bandit
        bestb.update(x)  # Update A and B based on whether A is True/False
        print("Parameters for Best bandit: a={}, b={}".format(bestb.a, bestb.b))
        
    plot(bandits)
    
if __name__ == '__main__':
    run_experiment(0.5, 0.5, 1000)  # Feel free to change these two probabilities as necessary