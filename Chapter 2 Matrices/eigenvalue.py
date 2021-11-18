import numpy as np 
from scipy.linalg import eigvalsh

A = np.array([[4,1,0],[1,4,1],[0,1,4]])

x = eigvalsh(A)

print(x)
