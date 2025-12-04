import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# Physical & numerical parameters
# -----------------------------
L = 1.0          # length of the string
N = 200          # number of spatial intervals -> N+1 grid points
dx = L / N

c = 1.0          # wave speed

# Choose dt to satisfy Courant condition: lambda = c*dt/dx <= 1
lambda_cfl = 0.9         # pick something < 1 for safety
dt = lambda_cfl * dx / c
tmax = 2.0               # total simulation time

# Gaussian pulse parameters (initial displacement)
x0 = 0.3 * L             # centre of pulse
sigma = 0.02             # width of pulse

# -----------------------------
# Grids
# -----------------------------
x = np.linspace(0.0, L, N + 1)
lambda2 = (c * dt / dx) ** 2

# -----------------------------
# Initial conditions
# -----------------------------
# u(x,0) = exp(- (x-x0)^2 / (2*sigma^2) )
u0 = np.exp(-0.5 * ((x - x0) / sigma) ** 2)

# u_t(x,0) = 0  (string released from rest)
# So first time step from Taylor expansion:
u1 = np.copy(u0)
# interior points j = 1,...,N-1
u1[1:-1] = (
    u0[1:-1]
    + 0.5 * lambda2 * (u0[2:] - 2 * u0[1:-1] + u0[:-2])
)

# Fixed boundary: u(0,t) = u(L,t) = 0
u0[0] = u0[-1] = 0.0
u1[0] = u1[-1] = 0.0

# Current and previous time levels
u_prev = u0      # u^0
u_curr = u1      # u^1

t = 0.0          # current time (corresponds to u_curr)

# -----------------------------
# Set up Matplotlib figure
# -----------------------------
fig, ax = plt.subplots()
line, = ax.plot(x, u_curr)
ax.set_xlim(0.0, L)
ax.set_ylim(-1.2, 1.2)
ax.set_xlabel("x")
ax.set_ylabel("u(x,t)")
ax.set_title("Wave equation – stable central difference")

# time label inside the axes
time_text = ax.text(0.02, 0.95, f"t = {t:.3f}",
                    transform=ax.transAxes,
                    ha="left", va="top")

plt.tight_layout()

# -----------------------------
# One time step of the scheme
# -----------------------------
def step():
    global u_prev, u_curr, t

    # allocate next time level
    u_next = np.zeros_like(u_curr)

    # apply the 3-level central-difference scheme at interior points
    # u_j^{n+1} = 2 u_j^n - u_j^{n-1}
    #             + lambda^2 (u_{j+1}^n - 2u_j^n + u_{j-1}^n)
    u_next[1:-1] = (
        2.0 * u_curr[1:-1]
        - u_prev[1:-1]
        + lambda2 * (u_curr[2:] - 2.0 * u_curr[1:-1] + u_curr[:-2])
    )

    # fixed boundary conditions
    u_next[0] = 0.0
    u_next[-1] = 0.0

    # shift time levels: n -> n-1, n+1 -> n
    u_prev, u_curr = u_curr, u_next

    # advance time
    t += dt


# -----------------------------
# Animation callbacks
# -----------------------------
def init():
    line.set_ydata(u_curr)
    time_text.set_text(f"t = {t:.3f}")
    return line, time_text


def update(frame):
    step()
    line.set_ydata(u_curr)
    time_text.set_text(f"t = {t:.3f}")
    return line, time_text


nframes = int(tmax / dt)

ani = FuncAnimation(
    fig,
    update,
    init_func=init,
    frames=nframes,
    interval=30,   # ms between frames
    blit=True
)

plt.show()
