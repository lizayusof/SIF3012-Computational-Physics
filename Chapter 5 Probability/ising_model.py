import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 20                # Lattice size (LxL)
T = 2.0               # Temperature
n_steps = 10000      # Number of Metropolis updates
J = 1.0               # Interaction strength
kB = 1.0              # Boltzmann constant

# Initialize the spin lattice randomly
spins = np.random.choice([-1, 1], size=(L, L))

# Save initial configuration
initial_spins = spins.copy()

# Function to compute energy of the system
def compute_energy(spins):
    energy = 0
    for i in range(L):
        for j in range(L):
            S = spins[i, j]
            neighbors = spins[(i+1)%L, j] + spins[i, (j+1)%L] + \
                        spins[(i-1)%L, j] + spins[i, (j-1)%L]
            energy += -J * S * neighbors
    return energy / 2.0  # Each pair counted twice

# Perform one Metropolis step (L*L spin updates)
def metropolis_step(spins, T):
    for _ in range(L * L):
        i, j = np.random.randint(0, L, size=2)
        S = spins[i, j]
        neighbors = spins[(i+1)%L, j] + spins[i, (j+1)%L] + \
                    spins[(i-1)%L, j] + spins[i, (j-1)%L]
        dE = 2 * J * S * neighbors
        if dE <= 0 or np.random.rand() < np.exp(-dE / (kB * T)):
            spins[i, j] *= -1

# Track energy over time
energies = []
steps = []

# Main simulation loop
for step in range(n_steps):
    metropolis_step(spins, T)
    if step % 1000 == 0:
        E = compute_energy(spins)
        energies.append(E)
        steps.append(step)

# --- Plot results ---

# 1. Initial Configuration
plt.figure(figsize=(5,5))
plt.imshow(initial_spins, cmap='gray', interpolation='nearest')
plt.title('Initial Configuration')
plt.axis('off')
plt.show()

# 2. Final Configuration
plt.figure(figsize=(5,5))
plt.imshow(spins, cmap='gray', interpolation='nearest')
plt.title(f'Final Configuration at T = {T}')
plt.axis('off')
plt.show()

# 3. Energy Plot
plt.figure(figsize=(7,4))
plt.plot(steps, energies, color='blue')
plt.xlabel('Monte Carlo Steps')
plt.ylabel('Energy')
plt.title('Energy vs Monte Carlo Steps')
plt.grid(True)
plt.tight_layout()
plt.show()
