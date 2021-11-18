import numpy as np
import matplotlib.pyplot as mpl
from scipy.integrate import simps

def fn(x):
    return np.sin(x) + 1

#integration using Simpson's integration
xs = np.linspace(0, 1.5 * np.pi, 100)
ys = fn(xs)
area = simps(ys, x=xs)

#integration using Monte Carlo
width = 1.5 * np.pi - 0  # The width from 0 to 1.5pi
samples = np.random.uniform(low=0, high=width, size=100000)
mc_area = fn(samples).mean() * width

#error using standard deviation
error = np.std(samples * width) / np.sqrt(samples.size)


mpl.plot(xs, ys, label="Function")
mpl.fill_between(xs, 0, ys, alpha=0.1)
mpl.text(1, 0.75, f"Area from Simps is {area:0.3f}", fontsize=12)
mpl.text(0.7, 0.5, f"Area from MC Integration is {mc_area:0.3f}±{error:0.3f}", fontsize=12)
mpl.xlabel("x")
mpl.legend()

mpl.show()
