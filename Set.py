import numpy as np
import tensorflow as tf

class Set(object):
    def __init__(self, d):
        self.d = d
    
    def distance_func(self, x):
        raise NotImplementedError
        
    def cond(self, x):
        raise NotImplementedError
    
    def sample(self, num_sample):
        raise NotImplementedError
        
    def spherical(self, radius, angles):
        a = np.concatenate((np.array([2*np.pi]),angles))
        cum_sines = np.sin(a)
        cum_sines[0] = 1
        cum_sines = np.cumprod(cum_sines)
        cosines = np.cos(a)
        cosines = np.roll(cosines,-1)
        return radius * cum_sines * cosines

class OpenBall(Set):
    def __init__(self, d, x0, r):
        super(OpenBall, self).__init__(d)
        self.r = r
        self.x0 = x0
        
    def distance_func(self, x):
        if np.linalg.norm(x-self.x0) == 0.0:
            return tf.constant(self.r,dtype = tf.float64)
        else:
            return np.sqrt(self.r**2 + np.linalg.norm(self.x0)**2) - np.linalg.norm(x)
    
    def cond(self, x):
        if np.linalg.norm(x-self.x0) < self.r:
            return True
        else:
            return False
        
    def sample(self):
        R = self.r * np.random.uniform(low = 0, high = 1)
        T = np.random.uniform(0, np.pi, size = self.d-1)
        T[-1] *= 2
        Z = self.spherical(R,T)
        return Z