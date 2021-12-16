#Python program to find largest eigenvalue & eigenvector
#Chapter 2 : Power Method
#SIF3012 Computational Physics


import numpy as np

A = np.array([[1,3],[2,2]])
z1 = [1,0]
z2 = [1,0]
n =  10

#start calculate the multiplication of power matrix & initial guess

for i in range(n):
  z1 = np.dot(A,z1)
  norm = np.linalg.norm(z1)
  #normalised form
  B = z1/norm
  print(B)

#calculating multiplication of power matrix & initial guess one power less than previous one

for i in range(n-1):
  z2 = np.dot(A,z2)

#calculate approximate largest eigenvector

C = np.min(abs(B))
eigvect = B/C


#calculate approximate dominating corresponding eigenvalue

y = eval('[1,0]')
eigvalue = np.dot(z1,y)/np.dot(z2,y)

print('Largest eigenvalue=', eigvalue)
print('Correspoding eigenvector is=',eigvect)
