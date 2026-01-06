import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns
sns.set()


T = 400
u = np.zeros((100,100))
u[30,20] = 1

fig, ax = plt.subplots(figsize=(20,10))
ax.axis('off')
plot = ax.contourf(u, cmap='jet')
def ans(f):
  global u, plot

def ans(f):
    global u, plot
    for j in range(100):
        for i in range(1,99):
            u[i,j] += (u[i+1,j] + u[i,j+1] + u[i-1,j] + u[i,j-1] - 4*u[i,j]) / 4

for c in plot.collections:
    c.remove()
plot = ax.contourf(u, cmap='jet')

anim = animation.FuncAnimation(fig, ans, frames=400)
plt.show()