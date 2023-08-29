import numpy as np
from scipy.special import gamma, betainc

class V01(object):
    def __init__(self, alpha, d):
        self.alpha = alpha
        self.d = d
        self.kappa = 2**(1-self.alpha)/self.alpha * gamma(self.d/2) / (gamma(self.d/2+self.alpha/2) * gamma(self.alpha/2))
        self.C = gamma(self.d/2 + self.alpha/2) * gamma(self.d/2 - self.alpha/2) / gamma(self.d/2)**2
    
    def spherical(self, radius, angles):
        a = np.concatenate((np.array([2*np.pi]),angles))
        cum_sines = np.sin(a)
        cum_sines[0] = 1
        cum_sines = np.cumprod(cum_sines)
        cosines = np.cos(a)
        cosines = np.roll(cosines,-1)
        return radius * cum_sines * cosines
    
    # Marginal radial density function of V_1(0,dy)
    def mu(self,r):
        I = r**(self.alpha-1) * (1-betainc(self.d/2-self.alpha/2,self.alpha/2,r**2)) 
        return self.alpha * self.C * I

    # Marginal radial cummulative function of V_1(0,dy)
    def F(self,r):
        return self.C * r**self.alpha * (1 - betainc(self.d/2-self.alpha/2,self.alpha/2,r**2)) + betainc(self.d/2,self.alpha/2,r**2)
    
    # Newton-Raphson method for the simulation of random variables
    def NewtonRaphson(self, F, f, U, r_init = 0.5, tol=10**-3):
        l = len(U)
        r = np.zeros(l)
        for i in range(l):
            r[i] = r_init
            error = abs(F(r[i])-U[i])
            while error > tol:
                r[i] = np.abs(r[i] - (F(r[i])-U[i]) / f(r[i]))
                error = abs(F(r[i])-U[i])
        return r
    
    # Generation of copies of V random variable with density f.
    def __call__(self, M = 1):
        # Radios según f
        U = np.random.uniform(size=M)
        R = self.NewtonRaphson(self.F, self.mu, U)
        while (R >= 1).any():
            U = np.random.uniform(size=M)
            R = self.NewtonRaphson(self.F, self.mu, U)
        # Ángulos
        T = np.pi * np.random.uniform(size = [M,self.d-1])
        T[:,-1] = 2 * T[:,-1]
        Z = np.zeros([M,self.d])
        for i in range(M):
            Z[i,:]= self.spherical(R[i],T[i,:])
        return Z