import numpy as np
import matplotlib.pyplot as plt

# SMPL subroutine equivalent
def SMPL(N, H, EK, U):
    H2 = 2.0 * H * H
    Q = EK * (1.0 + EK)
    for i in range(1, N - 1):
        x = i * H - 1.0
        P = 2.0 * (1.0 - x * x)
        P1 = -2.0 * x * H
        U[i + 1] = ((2.0 * P - H2 * Q) * U[i] + (P1 - P) * U[i - 1]) / (P1 + P)

# Main program body
N = 501
DL = 1.0e-6
H = 2.0 / (N - 1)

AK = 0.5
BK = 1.5
DK = BK - AK
EK = AK
ISTEP = 0

U = np.zeros(N)
U[0] = -1.0
U[1] = -1.0 + H

SMPL(N, H, EK, U)
F0 = U[-1] - 1.0

while abs(DK) > DL:
    EK = 0.5 * (AK + BK)
    U = np.zeros(N)
    U[0] = -1.0
    U[1] = -1.0 + H
    SMPL(N, H, EK, U)
    F1 = U[-1] - 1.0
    if F0 * F1 < 0:
        BK = EK
    else:
        AK = EK
        F0 = F1
    DK = BK - AK
    ISTEP += 1

# Print result
print(f"Steps: {ISTEP}, EK: {EK:.10f}, DK: {DK:.2e}, F1: {F1:.5e}")

# Plot result
x = np.linspace(-1, 1, N)
plt.plot(x, U, label=f"Eigenfunction for EK ≈ {EK:.6f}")
plt.title("Sturm-Liouville Eigenfunction (Legendre Equation)")
plt.xlabel("x")
plt.ylabel("U(x)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
