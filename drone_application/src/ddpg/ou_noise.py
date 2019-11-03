import numpy as np
import numpy.random as nr

class OUNoise():
    """docstring for OUNoise"""
    def __init__(self, action, eps, mu=0, theta=0.60, sigma=0.30):

        self.action = action
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.explore_noise = np.zeros([1, len(self.action[0])])
        self.eps = eps

    def noise(self):

        self.explore_noise[0][0] = max(self.eps, 0) * (self.theta * (self.mu - self.action[0][0]) + self.sigma * np.random.randn(1))
        self.explore_noise[0][1] = max(self.eps, 0) * (self.theta * (self.mu - self.action[0][1]) + self.sigma * np.random.randn(1))
        self.explore_noise[0][2] = max(self.eps, 0) * (self.theta * (self.mu - self.action[0][2]) + self.sigma * np.random.randn(1))

        return self.explore_noise

