import numpy as np
import tensorflow as tf
from PDE_setting import *
from scipy.special import gamma

class Example1(PDE_setting):
    def __init__(self,alpha,d):
        super(Example1, self).__init__(alpha,d)
    
    def source(self,x):
        return 2**self.alpha * gamma(self.alpha/2+self.d/2) * gamma(self.alpha/2+1) / gamma(self.d/2) * np.ones(len(x))
    
    def boundary(self,x):
        return 0
    
    def solution(self,x):
        normx2 = np.linalg.norm(x, axis=1)**2
        aux = np.zeros(normx2.shape)
        return tf.math.maximum(1-normx2,aux)**(self.alpha/2)