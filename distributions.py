import numpy as np

class distribution:
    def __init__(self, name):
        self.name = name

class gaussian(distribution):
    def __init__(self, mu=0., sigma=1.0):
        distribution.__init__(self, 'gaussian')
        self.mu = mu
        self.sigma = sigma
    def pdf(self, x):
        """probability density function"""
        return np.exp(- (x - self.mu)**2 / (2 * self.sigma**2))
