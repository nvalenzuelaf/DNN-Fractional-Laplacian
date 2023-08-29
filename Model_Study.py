import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class Model_Study(object):
    def __init__(self,S,radial):
        self.S = S
        self.u = S.u
        self.d = S.d
        self.alpha = S.alpha
        self.radial = radial
        self.time = None
    
    def fit_model(self,verbose=True):
        start_time = datetime.now()
        log = self.S.train(self.radial,verbose)
        self.time = datetime.now() - start_time
        return log

    def MSE(self,X):
        u = np.array([self.u(X).numpy()]).T
        Phi = self.S.model(X).numpy()
        resta = u - Phi
        return 1/len(X) * np.sum(resta**2)

    def Absolute_Error(self,X):
        u = np.array([self.u(X).numpy()]).T
        Phi = self.S.model(X).numpy()
        resta = np.abs(Phi - u)
        return 1/len(X) * np.sum(np.divide(resta,np.abs(u)))

    def Comparison2D(self,MC_plot = False,plot_show=True):
        x00 = np.linspace(0,1/np.sqrt(self.d) + 0.1,5000)
        Z00 = np.zeros([len(x00),self.d])
        for i in range(len(x00)):
            Z00[i,:] = x00[i]*np.ones(self.d)
        u_DNNN = []
        u_reall = []
        MC = []
        aux = self.S.model(Z00)
        for i in range(len(x00)):
            u_DNNN.append(aux[i].numpy()[0])
            xx = x00[i]*np.ones(self.d)
            if MC_plot:
                MC.append(self.S.model.MC(xx,self.S.M_init))
            u_reall.append(self.u(np.array([xx])))
        u_reall = np.nan_to_num(u_reall)
        plt.plot(x00,u_reall, label = 'Analytic Solution')
        plt.plot(x00,u_DNNN, label = 'Predicted Solution')
        if MC_plot:
            plt.plot(x00,MC, label = 'Monte Carlo Aprroximation')
        plt.title('Comparison of solutions for d='+str(self.d)+' and α='+str(self.alpha))
        if self.d==2:
            plt.xlabel('$x_1=x_2$')
        else:
            plt.xlabel('$x_1=...=x_{'+str(self.d)+'}$')

        plt.legend(loc = 'best')
        plt.grid()
        if plot_show:
            plt.show()

    def Comparison3D(self,plot_show=True):
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(1, 2, 2, projection='3d')

        # Data for a three-dimensional line
        xline = np.linspace(-1, 1, 1000)
        yline = np.linspace(-1, 1, 1000)
        X, Y = np.meshgrid(xline, yline)
        positions = np.column_stack([*[X.ravel()]*(self.d-1), Y.ravel()])
        U = self.S.model(positions).numpy().T[0]
        U = np.reshape(U,(len(xline),len(yline)))
        U = np.nan_to_num(U)
        surf = ax.plot_surface(X, Y, U, cmap='coolwarm',
                              linewidth=0, antialiased=False)
        ax.set_title('DNN solution for d= ' + str(self.d) +' and α='+str(self.alpha))
        if self.d == 2:
            ax.set_xlabel('$x_1$')
        elif self.d == 3:
            ax.set_xlabel('$x_1=x_2$')
        else:
            ax.set_xlabel('$x_1=...=x_{'+str(self.d-1)+'}$')
        ax.set_ylabel('$x_{'+str(self.d)+'}$')

        ax = fig.add_subplot(1, 2, 1, projection='3d')
        U = self.u(positions)
        U = np.reshape(U,(len(xline),len(yline)))
        U = np.nan_to_num(U)
        surf = ax.plot_surface(X, Y, U, cmap='coolwarm',
                              linewidth=0, antialiased=False)
        ax.set_title('Exact solution for d= ' + str(self.d) +' and α='+str(self.alpha))
        if self.d == 2:
            ax.set_xlabel('$x_1$')
        elif self.d == 3:
            ax.set_xlabel('$x_1=x_2$')
        else:
            ax.set_xlabel('$x_1=...=x_{'+str(self.d-1)+'}$')
        ax.set_ylabel('$x_{'+str(self.d)+'}$')
        plt.grid()
        if plot_show:
            plt.show()