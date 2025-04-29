#An example of solving the eigenvalue problem of the 1D Schrödinger equation 
# using the Numerov method and the secant method for root finding.
#Based on Tao Pang Computational Physics 2nd edition, 2004

import numpy as np
import matplotlib.pyplot as plt

# Constants
N = 501
M = 5
IMAX = 100
DL = 1.0e-6
H2M = 0.5
EA = 2.4
EB = 2.7
E0 = EA
E1 = EB
XL0 = -10.0
XR0 = 10.0
H = (XR0 - XL0) / (N - 1)
C = 1.0 / H2M

# Initialize arrays
UL = np.zeros(N)
UR = np.zeros(N)
QL = np.zeros(N)
QR = np.zeros(N)
R = np.zeros(N)

# Initial conditions
UL[0] = 0.0
UL[1] = 0.01
UR[0] = 0.0
UR[1] = 0.01

def VX(x):
    A = 1.0
    B = 4.0
    return 3.0 - A**2 * B * (B - 1.0) / (np.cosh(A*x)**2) / 2.0

def NMRV2(N, H, Q, S, U):
    G = H*H / 12.0
    for i in range(1, N-1):
        C0 = 1.0 + G * Q[i-1]
        C1 = 2.0 - 10.0 * G * Q[i]
        C2 = 1.0 + G * Q[i+1]
        D = G * (S[i+1] + S[i-1] + 10.0 * S[i])
        UTMP = C1 * U[i] - C0 * U[i-1] + D
        U[i+1] = UTMP / C2

def compute_F(E):
 #   Compute F(E) for a given trial energy E.
    for i in range(N):
        XL = XL0 + i * H
        XR = XR0 - i * H
        QL[i] = C * (E - VX(XL))
        QR[i] = C * (E - VX(XR))
        R[i] = 0.0

    # Find matching point
    IM = 0
    for i in range(N-1):
        if (QL[i] * QL[i+1] <= 0) and (QL[i] > 0):
            IM = i

    NL = IM + 1
    NR = N - IM + 2

    UL[0] = 0.0
    UL[1] = 0.01
    UR[0] = 0.0
    UR[1] = 0.01

    NMRV2(NL, H, QL, R, UL)
    NMRV2(NR, H, QR, R, UR)

    FACT = UR[NR-2] / UL[IM]
    UL[:NL] = FACT * UL[:NL]

    F = (UR[NR-1] + UL[NL-1] - UR[NR-3] - UL[NL-3]) / (2.0 * H * UR[NR-2])
    return F, IM

# Compute initial F0 and F1
F0, _ = compute_F(E0)
F1, _ = compute_F(E1)

ISTEP = 0

# Secant method loop
while abs(E1 - E0) > DL and ISTEP < IMAX:
    E2 = E1 - F1 * (E1 - E0) / (F1 - F0)

    # Update old values
    E0, E1 = E1, E2
    F0, F1 = F1, compute_F(E1)[0]

    ISTEP += 1

print(f"Converged after {ISTEP} steps.")
print(f"Eigenvalue E = {E1:.8f}")

# Now compute the final wavefunction
# (Same procedure as compute_F but now store the full wavefunction)

for i in range(N):
    XL = XL0 + i * H
    XR = XR0 - i * H
    QL[i] = C * (E1 - VX(XL))
    QR[i] = C * (E1 - VX(XR))
    R[i] = 0.0

IM = 0
for i in range(N-1):
    if (QL[i] * QL[i+1] <= 0) and (QL[i] > 0):
        IM = i

NL = IM + 1
NR = N - IM + 2

UL[0] = 0.0
UL[1] = 0.01
UR[0] = 0.0
UR[1] = 0.01

NMRV2(NL, H, QL, R, UL)
NMRV2(NR, H, QR, R, UR)

FACT = UR[NR-2] / UL[IM]
UL[:NL] = FACT * UL[:NL]

# Assemble full solution
for i in range(N):
    if i > IM:
        UL[i] = UR[N - i - 1]

# Normalize
SUM = np.sum(UL**2)
UL /= np.sqrt(H * SUM)

# X grid
x = np.linspace(XL0, XR0, N)
Vx = VX(x)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, UL, label='Wavefunction (normalized),$\psi(x)$', color='blue')
plt.plot(x, Vx, label='Potential V(x)', linestyle='dashed', color='black')
plt.axhline(0, color='black', lw=0.5)
plt.axvline(x=0, color='black', lw=0.5)
plt.text(-7.31,2.679,'$x_l$',fontsize=16)
plt.text(4.37,2.679,'$x_r$',fontsize=16)
plt.title('Wavefunction and Potential')
plt.xlabel('x')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
