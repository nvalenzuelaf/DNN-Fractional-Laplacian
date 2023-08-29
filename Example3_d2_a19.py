from Setting_example3 import *
from Solver import *
from Model_Study import *

lr = [5e-3,5e-3]
lr_bounds =[1000]
a = 1.9
d = 2
M = 300
n_it = 1000
valid_size = 1000
batch_size = 200

S = Solver(M, round(a,1), d, Example3, n_it, valid_size, batch_size, lr_bounds, lr)
print('Entrenamiento red neuronal para alpha = ' + str(round(a,1)))
MS = Model_Study(S,radial=False)
log = MS.fit_model()
MS.Comparison2D(MC_plot = False)
MS.Comparison3D()

print("Elapsed Time=",MS.time.total_seconds(),"[s]")