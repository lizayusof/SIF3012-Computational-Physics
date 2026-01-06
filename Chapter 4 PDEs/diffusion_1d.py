import numpy as np
from matplotlib import pyplot as plt

L = 100

x = np.linspace(0, 1, L)
u = np.zeros(L)

#Initial Condition
u[L//2] = 1

#Boundary Conditions
#This is not necessary to declare since u = zeros(L)
u[0] = 0
u[L-1] = 0

for t in range(100):
for i in range(1, L-1):
u[i] += (u[i+1] - 2*u[i] + u[i-1])/4

fig, ax = plt.subplots(figsize=(20,10))
ax.scatter(x,u, linewidth=15, c=u, cmap='jet')
plt.show()