import numpy as np
import matplotlib.pyplot as plt

class PDE_setting(object):
    def __init__(self,alpha,d):
        self.alpha = alpha
        self.d = d
    
    def source(self,x):
        raise NotImplementedError
    
    def boundary(self,x):
        raise NotImplementedError
    
    def solution(self,x):
        raise NotImplementedError
        
    def plot(self,verbose = True):
        # Data for a three-dimensional line
        xline = np.linspace(-1, 1, 100)
        yline = np.linspace(-1, 1, 100)
        X, Y = np.meshgrid(xline, yline)
        positions = np.column_stack([*[X.ravel()]*(self.d-1), Y.ravel()])
        U = self.solution(positions)
        U = np.reshape(U,(len(xline),len(yline)))
        #U = np.nan_to_num(U)
        fig = plt.figure(figsize=(7,6))
        ax = plt.axes(projection='3d')
        surf = ax.plot_surface(X, Y, U, cmap='coolwarm',
                               linewidth=0, antialiased=False)
        if verbose:
            ax.set_title('Exact solution for α='+str(self.alpha))
            if self.d == 2:
                ax.set_xlabel('$x_1$')
            elif self.d == 3:
                ax.set_xlabel('$x_1=x_2$')
            else:    
                ax.set_xlabel('$x_1=...=x_{'+str(self.d-1)+'}$')
            ax.set_ylabel('$x_{'+str(self.d)+'}$')
            ax.set_zlabel('$u(x)$')
        plt.show()