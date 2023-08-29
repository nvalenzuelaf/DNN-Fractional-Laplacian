import numpy as np
from Setting_example1 import *
from Solver import *
from Model_Study import *

def spherical(radius, angles):
    a = np.concatenate((np.array([2*np.pi]),angles))
    cum_sines = np.sin(a)
    cum_sines[0] = 1
    cum_sines = np.cumprod(cum_sines)
    cosines = np.cos(a)
    cosines = np.roll(cosines,-1)
    return radius * cum_sines * cosines

lr = [5e-3,5e-3]
lr_bounds =[1000]
d = 5
MM = [10,100,1000,2000]
valid_sizes = [10,100,1000,2000]
batch_sizes = [2,20,200,400]
alphas = [0.1,0.5,1.0,1.5,1.9]


n_it = 1000

R = np.random.uniform(low = 0, high = 1, size = 5000)
T = np.pi * np.random.uniform(size = [5000,d-1])
T[:,-1] = 2 * T[:,-1]
VALID_DATA = np.zeros([5000,d])
for i in range(5000):
    VALID_DATA[i,:]= spherical(R[i],T[i,:])

MSE = np.zeros([len(alphas),len(MM),len(valid_sizes)])
MRE = np.zeros([len(alphas),len(MM),len(valid_sizes)])
times = np.zeros([len(alphas),len(MM),len(valid_sizes)])
for i in range(len(alphas)):
	print('-'*10)
	print('DNN Aproximation for alpha =' + str(alphas[i]))
	for j in range(len(MM)):
		for k in range(len(valid_sizes)):
			print('-'*5)
			print('Training with M = '+str(MM[j])+' and P = '+str(valid_sizes[k]))
			S = Solver(MM[j], round(alphas[i],1), d, Example1, n_it, valid_sizes[k], batch_sizes[k], lr_bounds, lr)
			MS = Model_Study(S,radial=True)
			log = MS.fit_model(verbose=False)
			MSE[i,j,k] = MS.MSE(VALID_DATA)
			MRE[i,j,k] = MS.Absolute_Error(VALID_DATA)
			times[i,j,k] = MS.time.total_seconds()
			print("MSE: %.4e,    MRE: %.4e,     time: %.4e" % (MSE[i,j,k], MRE[i,j,k], times[i,j,k]))
	print('For alpha = '+str(alphas[i]))
	print(MSE[i,:,:])
	print(MRE[i,:,:])
	print(times[i,:,:])


# Errors' plots for different valid_sizes
for i in range(len(valid_sizes)):
	for j in range(len(alphas)):
		plt.loglog(MM,MSE[j,:,i],'-*',label='α='+str(round(alphas[j],1)))
	plt.title('Mean square error')
	plt.xlabel('Monte Carlo iterations')
	plt.ylabel('Error')
	plt.grid()
	plt.legend(loc='best')
	plt.show()

	for j in range(len(alphas)):
		plt.loglog(MM,MRE[j,:,i],'-*',label='α='+str(round(alphas[j],1)))
	plt.title('Mean relative error')
	plt.xlabel('Monte Carlo iterations')
	plt.ylabel('Error')
	plt.grid()
	plt.legend(loc='best')
	plt.show()

