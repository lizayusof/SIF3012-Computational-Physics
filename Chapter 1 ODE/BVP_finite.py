#Program to calculate linear BVPs using finite difference
#Created by Norhasliza Yusof, 5 November 2020, 1.09 pm
#This code is written for SIF3012, Computational Physics course

import numpy as np
import  scipy.sparse
from scipy.linalg import solve
import matplotlib.pyplot as mpl

#initialised all the parameter
N= 3
p = 1.0
q = 1.0
h = 1/(N+1)

#declare the function
def f(x):
  return np.sin(np.pi*x)

#set the boundary conditions
g0 = 1.0
g1 = 0.0

#Create the triadiagonal matrix A
#build the index
diagonals = np.zeros((3,N))
diagonals[0,:]=-(1.0+0.5*p*h)
diagonals[1,:]=(2+q*h**2)
diagonals[2,:]=-(1.0-0.5*p*h)
#transform index into triadiagonal matrix
A = scipy.sparse.spdiags(diagonals,[-1,0,1],N,N,format='csc')
#print A into array
A =  A.toarray()

#Create matrix b
b = np.zeros((3,1))
b[0,:]=(1.0+0.5*p*h)*g0+(h**2*f(0.25))
b[1,:]=(h**2)*f(0.5)
b[2,:]=(1.0-0.5*p*h)*g1 + (h**2*f(0.75))

#check your matrix A and b, uncomment if you want to check your result
#print(A)
#print(b)


#Solve the triadiagonal matrix
U =solve(A,b)
print('Final result')
print('U=', U)

#plot the matrix

#first create another matrix as x-scala for your result
#x is varying with h
x = np.zeros((3,1))
x[0,:]=0.25
x[1,:]=0.5
x[2,:]=0.75

#label your plot
mpl.xlabel('x')
mpl.ylabel('u(x)')

#plot matrix x and U
mpl.plot(x.T,U.T,'-rx')

#put the boundary condition in your plot
xx = [0.0,1.0]
yy = [1.0,0.0]
mpl.plot(xx,yy,'bo-')
mpl.xticks(np.arange(0,1,0.25))

#save figure
mpl.savefig('BVPfinitediff.png')

#show graph
mpl.show()

