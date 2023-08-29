import numpy as np
import tensorflow as tf
from WoS_process import *
from V01 import *
from PDE_setting import *

class DNN(tf.keras.Model):
    def __init__(self, num_hiddens):
        super(DNN, self).__init__()
        self.num_hiddens = num_hiddens
        self.Dense = [tf.keras.layers.Dense(num_hiddens[i], use_bias = True) for i in range(len(num_hiddens))]
        self.Out = tf.keras.layers.Dense(1, use_bias = True)
        
    def call(self,x,training = True):
        for i in range(len(self.Dense)):
            x = self.Dense[i](x)
            x = tf.nn.relu(x)
        x = self.Out(x)
        return x

class Model(tf.keras.Model):
    def __init__(self, alpha, d, PDE):
        super(Model,self).__init__()
        self.alpha = alpha
        self.d = d
        self.PhiDNN = DNN([110 for i in range(7)])
        self.WoS = WoS_process(self.alpha, self.d)
        self.V = V01(self.alpha, self.d)
        self.PDE = PDE(self.alpha,self.d)
        self.source = self.PDE.source
        self.boundary = self.PDE.boundary
        self.set = self.WoS.set
        
    def MC(self, x, M):
        if not self.set.cond(x):
            return self.boundary(x)
        Etot = 0
        for i in range(M):
            WoS,radius = self.WoS(x)
            N = len(radius)
            aux1 = radius**self.alpha * self.V.kappa #size N
            V = self.V()
            aux2 = WoS[:-1] + np.outer(radius,np.ones(self.d))*V #size Nxd
            Emu = np.sum(aux1 * self.source(aux2))
            Etot += Emu + self.boundary(WoS[-1])
        Etot = Etot/M
        return Etot   

    def call(self,X): 
        Phi = self.PhiDNN(X)
        return Phi