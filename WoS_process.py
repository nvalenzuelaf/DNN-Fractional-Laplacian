from Set import *
from scipy.special import gamma, betainc, betaincinv

class WoS_process(object):
    def __init__(self, alpha, d):
        self.alpha = alpha
        self.d = d
        self.set = OpenBall(self.d,np.zeros(self.d),1)
        
        self.C1 = np.sin(np.pi*self.alpha/2)/np.pi
        self.C2 = gamma(self.alpha/2) * gamma(1-self.alpha/2)

    # Marginal radial density function of X, when it exits the unitary ball.
    # Density does not depend on the dimension d.
    def f(self,r):
        return 2 * self.C1 / ((r**2-1)**(self.alpha/2) * r)

    # Marginal radial cummulative function of X, when it exits the unitary ball
    def F(self,r):
        return 1 - betainc(self.alpha/2,1-self.alpha/2,1/r**2)
    
    # Inverse function of F
    def Finv(self,u):
        y = 1 - u
        Iinv = betaincinv(self.alpha/2,1-self.alpha/2,y)
        return 1/np.sqrt(Iinv)   
        
    # Simulation of a copy of X exiting the unitary ball
    def Xballcopy(self):
        U = np.random.uniform()
        R = self.Finv(U)
        T = np.random.uniform(0, np.pi, size = self.d-1)
        T[-1] *= 2
        Z = self.set.spherical(R,T)
        return Z
    
    # Generation of a copy of the WoS process
    # Its output is rho: (N+1)xd sized vector that represents (rho_n)_{n=0}^{N} c IR^d
    # and radius: N sized vector that represents (r_n)_{n=1}^N c IR. 
    def __call__(self, x):
        rho = np.array([x])
        radius = []
        while self.set.cond(rho[-1]):
            rn = self.set.distance_func(rho[-1])
            radius.append(rn)
            Y = self.Xballcopy()
            aux = rho[-1] + rn * Y
            rho = np.vstack([rho,aux])
        return rho, np.array(radius)