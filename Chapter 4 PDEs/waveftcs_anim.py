import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# Parameters
# -----------------------------
L = 1.0          # length of the string
N = 1000          # number of spatial intervals -> N+1 grid points
dx = L / N
c = 1.0          # wave speed
dt = dx / c      # like in the notes: dt = dx/c
tmax = 1.0       # total evolution time

# Gaussian pulse parameters
x0 = 0.3 * L     # centre of pulse
s = 0.02         # width of pulse

# -----------------------------
# Grids and initial conditions
# -----------------------------
x = np.linspace(0.0, L, N + 1)

u = np.exp(-0.5 * ((x - x0) / s) ** 2)  # initial displacement u(0,x)
v = np.zeros_like(u)                    # auxiliary field
u_new = np.zeros_like(u)
v_new = np.zeros_like(v)

t = 0.0  # current time

# -----------------------------
# Set up Matplotlib figure
# -----------------------------
fig, ax = plt.subplots()
line, = ax.plot(x, u)
ax.set_xlim(0.0, L)
ax.set_ylim(-1.2, 1.2)
ax.set_xlabel("x (m)")
ax.set_ylabel("u")
ax.set_title("Wave equation – FTCS (unstable)")

# dynamic time label INSIDE the axes (top-left corner)
time_text = ax.text(0.02, 0.95, f"t = {t:.3f}",
                    transform=ax.transAxes,
                    ha="left", va="top")

plt.tight_layout()


# -----------------------------
# One FTCS update step
# -----------------------------
def step():
    global u, v, u_new, v_new, t

    # interior points j = 1,...,N-1
    v_new[1:-1] = v[1:-1] + 0.5 * dt * c * (u[2:] - u[:-2]) / dx
    u_new[1:-1] = u[1:-1] + 0.5 * dt * c * (v[2:] - v[:-2]) / dx

    # boundary conditions (fixed ends)
    u_new[0] = 0.0
    u_new[-1] = 0.0

    # "radiation" type condition for v (as in the notes)
    v_new[0] = v_new[1] + u_new[1]
    v_new[-1] = v_new[-2] - u_new[-2]

    # swap old and new
    u, u_new = u_new, u
    v, v_new = v_new, v

    t += dt


# -----------------------------
# Animation callbacks
# -----------------------------
def init():
    line.set_ydata(u)
    time_text.set_text(f"t = {t:.3f}")
    return line, time_text


def update(frame):
    step()
    line.set_ydata(u)
    time_text.set_text(f"t = {t:.3f}")
    return line, time_text


nframes = int(tmax / dt)

ani = FuncAnimation(
    fig,
    update,
    init_func=init,
    frames=nframes,
    interval=30,
    blit=True
)

plt.show()
