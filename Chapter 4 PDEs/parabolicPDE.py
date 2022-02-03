#Example for parabolic PDE

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as mpl

N = 100  # number of points to discretize
L = 0.5
X = np.linspace(0, L, N) # position along the rod
h = L / (N - 1)


def odefunc(u, t):
    dudt = np.zeros(X.shape)

    dudt[0] = 0 # constant at boundary condition
    dudt[-1] = 0

    # now for the internal nodes
    for i in range(1, N-1):
        dudt[i] = k * (u[i + 1] - 2*u[i] + u[i - 1]) / h**2

    return dudt

init = 100.0 * np.ones(X.shape) # initial temperature
init[0] = 350.0  # one boundary condition
init[-1] = 620.0 # the other boundary condition

