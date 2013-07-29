import numpy as np
import random

class distribution:
    def __init__(self, name):
        self.name = name

class gaussian(distribution):
    def __init__(self, mu=0., sigma=1.0):
        distribution.__init__(self, 'gaussian')
        self.mu = mu
        self.sigma = sigma
    def pdf(self, x):
        return np.exp(- (x - self.mu)**2 / (2 * self.sigma**2))
    def get_sample(self):
        return random.normalvariate(self.mu, self.sigma)
        
class uniform(distribution):
    def __init__(self, mu=0., sigma=1.0):
        distribution.__init__(self, 'uniform')
        self.mu = mu
        self.sigma = sigma
    def pdf(self, x):
        return np.where(np.logical_and(x-self.mu>=-self.sigma/2., x-self.mu<=self.sigma/2.), 
                        1./self.sigma, 0.)
    def get_sample(self):
        return random.uniform(self.mu-self.sigma/2., self.mu+self.sigma/2.)
