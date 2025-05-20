#This code is using Faddeev-Leverrier method to obtain eigenvalues of Jacobian matrix 
#from a damped harmonic oscillation system. From the output, we use the coefficients
#to obtain charaterisctic polynomial and later can be used to do stability analysis
# using IVP method. 

import numpy as np

def faddeev_leverrier(A):
    A = np.array(A, dtype=float)
    n = A.shape[0]
    Ak = A.copy()
    coeffs = [1.0]

    for k in range(1, n + 1):
        ak = -np.trace(Ak) / k
        coeffs.append(ak)
        if k != n:
            Ak += np.diag([ak] * n)
            Ak = A @ Ak

    return np.array(coeffs)

# Parameters for the oscillator
k = 2.0   # spring constant
c = 0.5   # damping coefficient

# Jacobian matrix
J = np.array([
    [0, 1],
    [-k, -c]
])

coeffs = faddeev_leverrier(J)

print("Characteristic polynomial coefficients:")
print(coeffs)

# This represents: λ^2 + cλ + k = 0
