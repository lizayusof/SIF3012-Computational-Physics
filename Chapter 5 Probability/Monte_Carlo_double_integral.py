import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import dblquad

# 1. Define the function f(x, y) = x+y
def f(x, y):
    return x + y

# Define limits based on the integral I = int_0^2 dx int_0^1 ... dy
x_min, x_max = 0.0, 1.0
y_min, y_max = 0.0, 1.0

# --- PART A: CALCULATIONS ---

# 1. Analytical/Exact Calculation (using SciPy dblquad)
# Note: dblquad expects func(y, x), so we wrap our function using lambda y, x: f(x, y)
# The outer integral is x (0 to 1), the inner is y (0 to 1)
exact_result, error_est = dblquad(lambda y, x: f(x, y), x_min, x_max, lambda x: y_min, lambda x: y_max)

# 2. Monte Carlo Integration
N = 10000  # Number of random samples

# Generate random points covering the rectangular floor
x_samples = np.random.uniform(low=x_min, high=x_max, size=N)
y_samples = np.random.uniform(low=y_min, high=y_max, size=N)

# Calculate heights at these random points
z_values = f(x_samples, y_samples)

# Calculate Volume = (Average Height) * (Area of Base)
domain_area = (x_max - x_min) * (y_max - y_min)
monte_carlo_result = z_values.mean() * domain_area

# Calculate Error (Standard Deviation)
monte_carlo_error = np.std(z_values * domain_area) / np.sqrt(N)

# Print results to console
print(f"--- Results ---")
print(f"Exact Result (dblquad):   {exact_result:.5f}")
print(f"Monte Carlo Result:       {monte_carlo_result:.5f} ± {monte_carlo_error:.5f}")


# --- PART B: 3D VISUALIZATION ---

# Create a grid for plotting the smooth surface
x_grid = np.linspace(x_min, x_max, 50)
y_grid = np.linspace(y_min, y_max, 50)
X, Y = np.meshgrid(x_grid, y_grid)
Z = f(X, Y)

# Setup figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)

# Add result text directly onto the plot
result_text = (
    f"Exact Volume: {exact_result:.3f}\n"
    f"Monte Carlo Volume: {monte_carlo_result:.3f}"
)
# Place text in 3D space (anchored somewhat arbitrarily in the visible area)
ax.text2D(0.05, 0.95, result_text, transform=ax.transAxes, fontsize=12,
          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Labels and Colorbar
ax.set_title(r'Double Integral Volume: $f(x, y) = x + y$')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
fig.colorbar(surf, shrink=0.5, aspect=10, label='f(x, y)')
plt.savefig('Monte_Carlo_double_integral.png')

plt.show()