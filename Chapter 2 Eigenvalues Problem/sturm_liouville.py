import numpy as np
import matplotlib.pyplot as plt

def smpl(N: int, H: float, EK: float) -> np.ndarray:
    """
    Implements the simplest algorithm for the Sturm-Liouville equation.
    """
    U = np.zeros(N)
    U[0] = -1.0
    U[1] = -1.0 + H
    H2 = 2.0 * H * H
    Q = EK * (1.0 + EK)

    for i in range(1, N - 1):
        x = i * H - 1.0
        P = 2.0 * (1.0 - x * x)
        P1 = -2.0 * x * H
        U[i + 1] = ((2.0 * P - H2 * Q) * U[i] + (P1 - P) * U[i - 1]) / (P1 + P)

    return U

def sturm_liouville_solver(N=501, DL=1.0e-6, a_init=0.5, b_init=1.5):
    """
    Solves the Legendre Sturm-Liouville problem using bisection and finite difference.
    """
    H = 2.0 / (N - 1)
    AK = a_init
    BK = b_init
    DK = BK - AK
    EK = AK
    ISTEP = 0

    U = smpl(N, H, EK)
    F0 = U[-1] - 1.0

    while abs(DK) > DL:
        EK = 0.5 * (AK + BK)
        U = smpl(N, H, EK)
        F1 = U[-1] - 1.0
        if F0 * F1 < 0:
            BK = EK
        else:
            AK = EK
            F0 = F1
        DK = BK - AK
        ISTEP += 1

    x = np.linspace(-1, 1, N)
    return ISTEP, EK, DK, F1, U, x

# Run the solver
ISTEP, EK, DK, F1, U, x = sturm_liouville_solver()

# Plot the eigenfunction
plt.figure(figsize=(8, 5))
plt.plot(x, U, label=f"Eigenfunction for EK ≈ {EK:.6f}")
plt.title("Sturm-Liouville Eigenfunction (Legendre Equation)")
plt.xlabel("x")
plt.ylabel("U(x)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

