# import numpy.typing
# import numpy             as np
# import matplotlib.pyplot as plt

# # Constants
# N = 2
# omega = 1.8
# nu = 1/4 * (1/omega - 1/2)
# A = 0.05
# beta = 0.5
# rho0 = 100 # Average density
# Ndir = 9
# idx = np.arange(Ndir)

# num_timesteps = 10000


# # Middle circle
# X, Y = np.meshgrid(range(N), range(N))
# circle = (X - (N-1)/2)**2 + (Y - (N-1)/2)**2 < (N/10)**2

# # Initialize lattice
# Ni   = np.ones((N, N, Ndir), dtype=np.float32)
# rho_ = np.sum(Ni, axis=2)
# for i in range(Ndir):
#     Ni[:, :, i] *= rho0 / rho_
# Ni[circle, :] *= 2

# # Create plot for initial condition and final distribution
# fig1, ax1 = plt.subplots(nrows=1, ncols=2, layout='constrained')
# ax1[0].set_title("Initial Condition")
# im1 = ax1[0].imshow(Ni.sum(axis=2))

# # Create plot for visualizing the distribtuion at several time instants
# fig2, ax2 = plt.subplots(nrows=2, ncols=2, layout='constrained')#, layout='constrained')
# ax2 = ax2.flatten()
# ax2[0].set_title("Initial Condition")
# im = ax2[0].imshow(Ni.sum(axis=2))
# plot_times = [int(num_timesteps/3), int(num_timesteps*2/3), num_timesteps-1]
# plot_idx = 1


# # D2Q9 directions and weights
# ex = np.array([0, 1, -1, 0, 0, 1, -1, 1, -1])
# ey = np.array([0, 0, 0, 1, -1, 1, 1, -1, -1])
# weights = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

# Ni_eq = np.zeros(Ni.shape, dtype=np.float32)

# # Timeloop
# for t in range(num_timesteps):
#     print(f"t = {t}")

#     # Calculate fluid variables
#     rho = np.sum(Ni, axis=2) # Density
#     ux  = np.sum(Ni*ex, axis=2) / rho # x velocity
#     uy  = np.sum(Ni*ey, axis=2) / rho # y velocity
    
#     # Compute Ni_eq
#     for i, cx, cy, wi in zip(idx, ex, ey, weights):
#         Qi_term = 3/2 * (cx*ux + cy*uy)**2 - 1/2 * (ux**2 + uy**2)
#         Ni_eq[:, :, i] = rho*wi*(1 + 3*(cx*ux + cy*uy) + 3*Qi_term)

#     # Collision step
#     Ni += - omega*(Ni - Ni_eq)

#     # Streaming step
#     for i, cx, cy in zip(idx, ex, ey):
#         Ni[:, :, i] = np.roll(Ni[:, :, i], cx, axis=1)
#         Ni[:, :, i] = np.roll(Ni[:, :, i], cy, axis=0)

#     if t in plot_times:
#         # Plot the distribution
#         ax2[plot_idx].set_title(f"Time t = {t}")
#         im = ax2[plot_idx].imshow(Ni.sum(axis=2))
#         plot_idx += 1
    
# fig2.figure.colorbar(im, ax=ax2, shrink=0.5, location='bottom')

# ax1[1].set_title("Final time")
# im2 = ax1[1].imshow(Ni.sum(axis=2))
# fig1.figure.colorbar(im2, ax=ax1, shrink=0.75, location='bottom')
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
nx, ny = 100, 100  # Grid size
rho0 = 1.0         # Initial density
tau = 0.6          # Relaxation time
omega = 1.0 / tau  # Relaxation parameter

# Lattice weights and velocities for D2Q9 lattice
w = [4/9] + [1/9] * 4 + [1/36] * 4
dx = np.array([0, 1, -1, 0, 0, 1, -1, 1, -1])
dy = np.array([0, 0, 0, 1, -1, 1, 1, -1, -1])

# Arrays to store distribution functions
feq = np.zeros((9, nx, ny))
f = np.zeros((9, nx, ny))

# Additional variable for color gradient
phi = np.zeros((nx, ny))

# Initialization
rho = np.ones((nx, ny)) * rho0
ux, uy = np.zeros((nx, ny)), np.zeros((nx, ny))

# Set up initial conditions
rho[:, :] = rho0
ux[:, :] = np.random.rand(nx, ny)  # Initial random color gradient

# Function to calculate equilibrium distribution
def equilibrium(rho, ux, uy):
    usqr = 3/2 * (ux**2 + uy**2)
    feq[0, :, :] = w[0] * rho * (1 - usqr)
    for i in range(1, 9):
        cu = 3 * (dx[i] * ux + dy[i] * uy)
        feq[i, :, :] = w[i] * rho * (1 + cu + 0.5 * cu**2 - usqr)

# Function to perform LBM collision step
def collide():
    global f, feq, rho, ux, uy, phi
    equilibrium(rho, ux, uy)
    for i in range(9):
        f[i, :, :] = f[i, :, :] - (f[i, :, :] - feq[i, :, :]) / tau

# Function to perform LBM streaming step
def stream():
    global f
    for i in range(9):
        f[i, :, :] = np.roll(np.roll(f[i, :, :], dx[i], axis=0), dy[i], axis=1)

# Function to update macroscopic variables and color gradient
def update_macroscopic():
    global rho, ux, uy, phi
    rho = np.sum(f, axis=0)
    ux = (np.sum(f[1:4, :, :], axis=0) + np.sum(f[5:8, :, :], axis=0) + 2*np.sum(f[2:4, :, :], axis=0) - 2*np.sum(f[6:8, :, :], axis=0)) / rho
    uy = (np.sum(f[3:6, :, :], axis=0) + np.sum(f[7:9, :, :], axis=0) + 2*np.sum(f[2:4, :, :], axis=0) - 2*np.sum(f[0:2, :, :], axis=0)) / rho

    # Update color gradient (phi) based on fluid velocity
    phi = phi - 0.01 * np.sqrt(ux**2 + uy**2)

# Function to perform the LBM simulation step
def lbm_step():
    collide()
    stream()
    update_macroscopic()

# Animation update function
def update(frame):
    lbm_step()
    im.set_array(phi)
    return im,


for t in range(1000):
    lbm_step()
    plt.imshow(np.sqrt(ux**2 + uy**2), cmap='viridis')
    plt.colorbar()
    plt.show()

# # Create figure and axis
# fig, ax = plt.subplots()
# im = ax.imshow(phi, cmap='viridis', animated=True)

# # Create animation
# ani = animation.FuncAnimation(fig, update, frames=200, interval=50, blit=True)
# plt.show()