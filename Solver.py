import numpy as np
import tensorflow as tf
from PDE_setting import *
from Model import *

class Solver(object):
    def __init__(self, M_init, alpha, d, PDE, num_iterations, valid_size, batch_size, lr_bounds, lr):
        self.alpha = alpha
        self.d = d
        self.PDE = PDE(self.alpha,self.d)
        self.source = self.PDE.source
        self.boundary = self.PDE.boundary
        self.u = self.PDE.solution

        self.M_init = M_init
        self.num_iterations = num_iterations
        self.valid_size = valid_size
        self.batch_size = batch_size

        self.model = Model(self.alpha, self.d, PDE)
        self.set = self.model.set

        R = np.random.uniform(low = 0, high = 1, size = 5000)
        T = np.pi * np.random.uniform(size = [5000,self.d-1])
        T[:,-1] = 2 * T[:,-1]
        self.VALID_DATA = np.zeros([5000,self.d])
        for i in range(5000):
            self.VALID_DATA[i,:]= self.set.spherical(R[i],T[i,:])

        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            lr_bounds, lr)
        
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule) #for mac
        #self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule) #fow windows
    
    def MSE(self):
        u = np.array([self.u(self.VALID_DATA)]).T
        Phi = self.model(self.VALID_DATA).numpy()
        resta = u - Phi
        return 1/len(self.VALID_DATA) * np.sum(resta**2)

    def Absolute_Error(self):
        u = np.array([self.u(self.VALID_DATA)]).T
        Phi = self.model(self.VALID_DATA).numpy()
        resta = np.abs(Phi - u)
        return 1/len(self.VALID_DATA) * np.sum(np.divide(resta,np.abs(u)))
    
    def train(self,radial = False,verbose = True):
        training_history = []
        R1 = np.random.uniform(low = 0, high = 1, size = self.valid_size)
        R2 = np.random.uniform(low = 1, high = 1.5, size = self.valid_size)
        T = np.pi * np.random.uniform(size = [self.valid_size,self.d-1])
        T[:,-1] = 2 * T[:,-1]
        valid_data = np.zeros([self.valid_size,self.d])
        for i in range(self.valid_size):
            R = np.random.choice([R1[i],R2[i]],1,p=[0.7,0.3])[0]
            valid_data[i,:]= self.set.spherical(R,T[i,:])
        MC = np.zeros((self.valid_size,1))
        for j in range(self.valid_size):
            MC[j] = self.model.MC(valid_data[j],self.M_init)
        
        for step in range(self.num_iterations+1):
            if step % 100 == 0:
                Phi = self.model(valid_data)
                if radial:
                    Phiminus = self.model(-valid_data)
                    delta = tf.square(tf.subtract(MC,Phi))/2 + tf.square(tf.subtract(Phi,Phiminus))/2
                else:
                    delta = tf.square(tf.subtract(MC,Phi))
                loss = tf.reduce_mean(delta)
                mse = self.MSE() 
                mae = self.Absolute_Error()
                if  np.isnan(loss):
                    break
                training_history.append([step, loss, mse, mae])
                
                if verbose:
                    print("step: %5u,    loss: %.4e,     MSE: %.4e,     MAE: %.4e" % (step, loss, mse, mae))
            
            index = np.random.choice(self.valid_size,size = self.batch_size, replace = False)
            batch_data = valid_data[index]
            batch_data_true = MC[index]
            self.train_step(batch_data,batch_data_true,radial)
        self.loss = loss
        return np.array(training_history), Phi

    def loss_fn(self,valid_data,valid_data_true,radial):
        Phi = self.model(valid_data)
        if radial:
            Phiminus = self.model(-valid_data)
            delta = tf.square(tf.subtract(valid_data_true,Phi))/2 + tf.square(tf.subtract(Phi,Phiminus))/2
        else:
            delta = tf.square(tf.subtract(valid_data_true,Phi))
        loss = tf.reduce_mean(delta)
        return loss, Phi
    
    def grad(self,valid_data,valid_data_true,radial):
        with tf.GradientTape(persistent=True) as tape:
            loss, Phi = self.loss_fn(valid_data,valid_data_true,radial)
            
        grad = tape.gradient(loss, self.model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        del tape
        return grad, Phi

    def train_step(self,valid_data,valid_data_true,radial):
        grad, Phi = self.grad(valid_data,valid_data_true,radial)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))