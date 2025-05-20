import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the system: damped harmonic oscillator
def damped_oscillator(t, Y, k=2.0, c=0.5):
    x, y = Y
    dxdt = y
    dydt = -k * x - c * y
    return [dxdt, dydt]

# Time span and initial conditions
t_span = (0, 20)
t_eval = np.linspace(t_span[0], t_span[1], 400)

# Generate phase portrait for multiple initial conditions
initial_conditions = [
    [2.0, 0.0],
    [0.0, 2.0],
    [-2.0, 0.0],
    [0.0, -2.0],
    [1.5, 1.5],
    [-1.5, -1.5],
    [1.0, -1.0],
    [-1.0, 1.0]
]

plt.figure(figsize=(8, 6))

for y0 in initial_conditions:
    sol = solve_ivp(damped_oscillator, t_span, y0, t_eval=t_eval, args=(2.0, 0.5))
    plt.plot(sol.y[0], sol.y[1], label=f'x₀={y0[0]}, y₀={y0[1]}')

plt.title("Phase Portrait of a Damped Harmonic Oscillator")
plt.xlabel("x (position)")
plt.ylabel("y (velocity)")
plt.axhline(0, color='gray', lw=0.5)
plt.axvline(0, color='gray', lw=0.5)
plt.grid(True)
plt.legend(loc="upper right", fontsize="small")
plt.tight_layout()
plt.show()
