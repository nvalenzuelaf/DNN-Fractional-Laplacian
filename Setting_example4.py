import numpy as np
import tensorflow as tf
from PDE_setting import *
from scipy.special import gamma

class Example4(PDE_setting):
    def __init__(self,alpha,d):
        super(Example4, self).__init__(alpha,d)
    
    def source(self,x):
        return np.zeros(len(x))
    
    def boundary(self,x):
        return np.sum(x)
    
    def solution(self,x):
        return np.sum(x,axis=1)