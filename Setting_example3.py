import numpy as np
import tensorflow as tf
from PDE_setting import *
from scipy.special import gamma

class Example3(PDE_setting):
    def __init__(self,alpha,d):
        super(Example3, self).__init__(alpha,d)
    
    def source(self,x):
        return np.zeros(len(x))
    
    def boundary(self,x):
        y = np.zeros(self.d)
        y[0] = 2
        C = gamma(self.d/2-self.alpha/2) / (2**self.alpha * np.pi**(self.d/2) * gamma(self.alpha/2))
        return C * np.linalg.norm(x-y)**(self.alpha-self.d) 
    
    def solution(self,x):
        y = np.zeros(self.d)
        y[0] = 2
        C = gamma(self.d/2-self.alpha/2) / (2**self.alpha * np.pi**(self.d/2) * gamma(self.alpha/2))
        return C * tf.norm(x-y,axis=1)**(self.alpha-self.d) 