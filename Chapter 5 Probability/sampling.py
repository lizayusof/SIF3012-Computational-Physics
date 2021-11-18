import numpy as np
import matplotlib.pyplot as mpl
from scipy.integrate import simps


# MC integration here
samples_2 = np.random.normal(size=1000)
fn_2 = 1 + samples_2 ** 2
area_2 = fn_2.mean()
error_2 = np.std(fn_2) / np.sqrt(fn_2.size)

# Simps integration here
def fn2(xs):
    return (1 + xs**2) * np.exp(-(xs**2)/2) / np.sqrt(2 * np.pi)
xs = np.linspace(-5, 5, 200)
ys = fn2(xs)
area_simps = simps(ys, x=xs)

# And of course, plotting here
mpl.plot(xs, ys, label="Function", lw=3)
mpl.fill_between(xs, 0, ys, alpha=0.1)
mpl.text(-4.8, 0.5, f"MC Area={area_2:0.2f}±{error_2:0.2f}", fontsize=12)
mpl.text(-4.8, 0.43, f"Simps Area={area_simps:0.2f}", fontsize=12)
mpl.plot((samples_2, samples_2), ([0 for i in samples_2], [fn2(i) for i in samples_2]), 
         c='#1c93e8', lw=0.2, ls='-', zorder=-1, alpha=0.5)
mpl.xlabel("x")
mpl.legend()
mpl.show()
