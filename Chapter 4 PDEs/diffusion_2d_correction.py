import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

nx, ny = 100, 100
T = 400

u = np.zeros((nx, ny))
u[30, 20] = 1.0     # initial hot spot

fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('off')
plot = ax.contourf(u, cmap='jet')

def ans(frame):
    global u, plot

    # make a copy so we use only values from the old time level
    u_new = u.copy()

    # update only interior points: 1 .. nx-2, 1 .. ny-2
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            u_new[i, j] += (u[i+1, j] + u[i-1, j] +
                            u[i, j+1] + u[i, j-1] -
                            4*u[i, j]) / 4.0

    u[:] = u_new  # copy back

    # redraw
    for c in plot.collections:
        c.remove()
    plot = ax.contourf(u, cmap='jet')
    return plot

anim = animation.FuncAnimation(fig, ans, frames=T)
plt.show()
