import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

#we solving f(x)=x^2

def f(x):
    return x**2 

a = 0.0
b = 1.0
n = 10000

exact = 1/3

x = np.linspace(a,b,n)

#test using Mid-Point rule
M = (b-a)*f(a+b/2)
print('Midpoint result=',M)

#test using Trapeziod rule
T = (0.5*(b-a))*(f(a)+f(b))
print('Trapeziod results=',T)

#perform Monte Carlo calculation
#integration using Monte Carlo
area = simps(f(x), x)
print('Simpson result=',area)
width = b-a
samples = np.random.uniform(low=0, high=width, size=10000)
monte_carlo = f(samples).mean()* width
print('Monte Carlo=',monte_carlo)

#error using standard deviation
error = np.std(samples * width) / np.sqrt(samples.size)

plt.plot(x, f(x), label="f(x)=sin(x)")
plt.fill_between(x, 0, f(x), alpha=0.1)
plt.text(0.05, 0.70, f"Result from Mid Point is {M:0.3f}", fontsize=10)
plt.text(0.05, 0.65, f"Result from Trapeziod is {T:0.3f}", fontsize=10)
plt.text(0.05, 0.6, f"Area from Simps is {area:0.3f}", fontsize=10)
plt.text(0.05, 0.55, f"Area from MC Integration is {monte_carlo:0.3f}±{error:0.3f}", fontsize=10)


plt.legend()
plt.savefig('Monte-Carlo_1D.png')
plt.show()


