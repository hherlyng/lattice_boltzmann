import matplotlib
import numpy             as np
import matplotlib.pyplot as plt

# Constants
N = 5
omega = 1.8
A = 0.05
beta = 0.5
Ndir = 9
k   = 2*np.pi/N # Frequency
idx = np.arange(Ndir)
num_timesteps = 100

t_plot   = np.arange(num_timesteps)
sigma    = np.array([0]*num_timesteps, dtype=np.float64)
pressure_diff = np.copy(sigma)

# D2Q9 directions and weights
ex = np.array([0, 1, -1, 0, 0, 1, -1, 1, -1])
ey = np.array([0, 0, 0, 1, -1, 1, 1, -1, -1])
weights = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

# Middle circle
x = np.arange(N)
y = np.arange(N)
X, Y = np.meshgrid(x, y)
circle  = (X - (N-1)/2)**2 + (Y - (N-1)/2)**2 <= (N/10)**2
outside = (X - (N-1)/2)**2 + (Y - (N-1)/2)**2 >  (N/10)**2

# Initialize lattice
Ri   = np.zeros((N, N, Ndir), dtype=np.float64)
Bi   = np.zeros((N, N, Ndir), dtype=np.float64)

# Red fluid circle in center, blue fluid outside
Ri[circle,  :] = 1
Bi[outside, :] = 1

# Red fluid at bottom, blue fluid on top
# Ri[np.where(Y >= N/2 + 10*N/150*np.sin(k*X))] = 1.5
# Bi[np.where(Y < N/2 + 10*N/150*np.sin(k*X))] = 1

# Total population
Ni = Ri+Bi

# Create plot for initial condition and final distribution
# fig1, ax1 = plt.subplots(nrows=1, ncols=2, layout='constrained')
# ax1[0].set_title("Initial Condition")
# im1 = ax1[0].imshow(Ri.sum(axis=2))
# im2 = ax1[0].imshow(Bi.sum(axis=2))
# plt.show()

# Create plot for visualizing the distribtuion at different times
fig2, ax2 = plt.subplots(nrows=2, ncols=2, figsize=(12, 9), layout='constrained')
ax2 = ax2.flatten()
ax2[0].set_title("Initial Condition")
RGB = np.zeros((N, N, 3))
R_summed = Ri.sum(axis=2)
B_summed = Bi.sum(axis=2)
RGB[:, :, 0] = R_summed
RGB[:, :, 2] = B_summed
im = ax2[0].imshow(RGB)
plot_times = [int(num_timesteps/3), int(num_timesteps*2/3), num_timesteps-1]
plot_idx = 1

# Create plot for visualizing the centerline pressure at different times
fig3, ax3 = plt.subplots(nrows=3, ncols=1, figsize=(12, 9), sharex=True, layout='constrained')

## Utility functions

def calc_color_gradient(Ri, Bi):
    # Color gradient components
    fx = np.zeros((N, N), dtype=np.float64)
    fy = np.zeros((N, N), dtype=np.float64)

    for i, cx, cy in zip(idx, ex, ey):

        R_temp = np.copy(Ri)
        B_temp = np.copy(Bi)

        # Red fluid at x + ci
        R_temp[:, :, i] = np.roll(R_temp[:, :, i], shift=(cx, cy), axis=(1, 0))

        # Blue fluid at x + ci
        B_temp[:, :, i] = np.roll(B_temp[:, :, i], shift=(cx, cy), axis=(1, 0))

        inner_sum = R_temp.sum(axis=2) - B_temp.sum(axis=2)

        fx += cx*inner_sum
        fy += cy*inner_sum

    return fx, fy

def antidiffusive_recoloring(Ni_prime, rho, Ri, Bi, cos_phi):
    
    R = Ri.sum(axis=2)
    B = Bi.sum(axis=2)

    R_out = np.zeros_like(Ri)
    B_out = np.zeros_like(Bi)

    for i, wi in zip(idx, weights):
        R_out[:, :, i] = R/(R+B) * Ni_prime[:, :, i] + beta * R*B/(R+B)**2 * rho * wi * cos_phi[:, :, i]
        B_out[:, :, i] = B/(R+B) * Ni_prime[:, :, i] - beta * R*B/(R+B)**2 * rho * wi * cos_phi[:, :, i]
    
    return R_out, B_out

Ni_eq      = np.zeros_like(Ni)
color_term = np.zeros_like(Ni)
cos_phi    = np.zeros_like(Ni)

# Timeloop
for t in range(num_timesteps):
    print(f"t = {t}")

    # Calculate fluid variables
    rho = np.sum(Ni, axis=2) # Density
    ux  = np.sum(Ni*ex, axis=2) / rho # x velocity
    uy  = np.sum(Ni*ey, axis=2) / rho # y velocity

    # Compute color gradient
    fx, fy = calc_color_gradient(Ri, Bi)
    fx_hat, fy_hat = np.copy(-1*fx), np.copy(-1*fy)
    f_norm = np.sqrt(fx**2 + fy**2)

    # Normalize the color gradient by the norm
    fnorm_nzero_idx = np.invert(np.isclose(f_norm, 0.0))
    fx_hat[fnorm_nzero_idx] /= f_norm[fnorm_nzero_idx]
    fy_hat[fnorm_nzero_idx] /= f_norm[fnorm_nzero_idx]
    
    # Compute Ni_eq and the color gradient term
    for i, cx, cy, wi in zip(idx, ex, ey, weights):
        # Equilibrium distribution
        Qi_term_u = 3/2*(cx*ux + cy*uy)**2 - 1/2*(ux**2 + uy**2)
        Ni_eq[:, :, i] = rho*wi*(1 + 3*(cx*ux + cy*uy) + 3*Qi_term_u)

        # Color gradient
        Qi_term_f = 3/2*(cx*fx_hat + cy*fy_hat)**2 - 1/2*(fx_hat**2 + fy_hat**2)
        color_term[:, :, i] = A*f_norm*wi*Qi_term_f
    
    # Check that sum(Ni_eq) over all directions i equals rho at every lattice site
    assert np.allclose(Ni_eq.sum(axis=2), rho)
    from IPython import embed;embed()
    # Collision step
    Ni = Ni - omega*(Ni - Ni_eq) + color_term

    # Calculate cosine of the angle phi between the color gradient and the directions ci
    for i, cx, cy in zip(idx, ex, ey):
        if i == 0:
            cx_hat = cx
            cy_hat = cy
        else:
            cx_hat = cx / np.sqrt(cx**2+cy**2)
            cy_hat = cy / np.sqrt(cx**2+cy**2)

        cos_phi[:, :, i] = fx_hat*cx_hat + fy_hat*cy_hat
    # from IPython import embed;embed(header="first")
    # Antidiffusive recoloring step
    Ri, Bi = antidiffusive_recoloring(Ni_prime=Ni, rho=rho, Ri=Ri, Bi=Bi, cos_phi=cos_phi)
    # from IPython import embed;embed(header="second")
    # Propagation step
    for i, cx, cy in zip(idx, ex, ey):
        # Red fluid
        Ri[:, :, i] = np.roll(Ri[:, :, i], shift=(cx, cy), axis=(1, 0))
        
        # Blue fluid
        Bi[:, :, i] = np.roll(Bi[:, :, i], shift=(cx, cy), axis=(1, 0))
        
    Ni = Ri+Bi
    
    # Calculate pressure and surface tension
    P = rho/3
    idx_center  = int((N-1)/2)
    idx_outside = 0
    P_center  = P[idx_center, idx_center]
    P_outside = P[idx_outside, idx_outside]
    delta_P = P_center - P_outside
    dist = np.sqrt((X[idx_outside, idx_outside] - X[idx_center, idx_center])**2 + (Y[idx_outside, idx_outside] - Y[idx_center, idx_center])**2)
    sigma[t] = delta_P*dist
    pressure_diff[t] = delta_P

    # P_top    = P[5, int(N/2)]
    # P_bottom = P[N-5, int(N/2)]
    # delta_P = P_top - P_bottom
    # dist = np.sqrt((X[5, int(N/2)] - X[N-5, int(N/2)])**2 + (Y[5, int(N/2)] - Y[N-5, int(N/2)])**2)
    # sigma[t] = delta_P*dist
    # pressure_diff[t] = delta_P
    
    # if t % 1000 == 0:
    #     P_half = P[int(N/2), :]
    #     plt.plot(X[int(N/2), :], P_half)
    #     plt.show()


    # P_plot = P[idx_center, :]
    # plt.plot(X[idx_center, :], P_plot)
    # plt.show()
    #plt.imshow(Ri.sum(axis=2))# make a color map of fixed colors
    # cmap = matplotlib.colors.ListedColormap(['blue', 'red'])
    # bounds=[0, 1]
    # norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    # # tell imshow about color map so that only set colors are used
    # bool_grid = np.zeros_like(Ri.sum(axis=2))
    # indices = np.where(np.invert(np.isclose(Ri.sum(axis=2), 0.0)))
    
    # bool_grid[indices] = 1
    # img = plt.imshow(bool_grid, interpolation='nearest', origin='lower',
    #                     cmap=cmap, norm=norm)

    # # make a color bar
    # plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds, ticks=[0, 1])
    # res = Ri.sum(axis=2) - Bi.sum(axis=2)
    # pl  = np.zeros_like(res)
    # pl[res>0] = 1 # Red
    # plt.imshow(pl)
    # plt.show()
    # from IPython import embed;embed()

    # Create RGB array for plotting
    # RGB = np.zeros((N, N, 3))
    # R_summed = Ri.sum(axis=2)
    # if np.max(R_summed) != 0:
    #     R_summed = R_summed[:, :]*1.0/np.max(R_summed)

    # B_summed = Bi.sum(axis=2)
    # max_B = np.max(B_summed)
    # if np.max(B_summed) != 0:
    #     B_summed = B_summed[:, :]*1.0/np.max(B_summed)
    
    # RGB[:, :, 0] = R_summed
    # RGB[:, :, 2] = B_summed
    # # from IPython import embed;embed()
    # plt.imshow(RGB)
    # plt.show()

    # Visualize
    if t in plot_times:
        # Plot the distribution
        ax2[plot_idx].set_title(f"Time t = {t}")

        # Create RGB array for plotting
        RGB = np.zeros((N, N, 3))
        R_summed = Ri.sum(axis=2)
        if np.max(R_summed) != 0:
            R_summed = R_summed[:, :]*1.0/np.max(R_summed)

        B_summed = Bi.sum(axis=2)
        max_B = np.max(B_summed)
        if np.max(B_summed) != 0:
            B_summed = B_summed[:, :]*1.0/np.max(B_summed)
        
        RGB[:, :, 0] = R_summed
        RGB[:, :, 2] = B_summed
        im = ax2[plot_idx].imshow(RGB)

        # Plot the centerline pressure
        P_half = P[int(N/2), :]
        ax3[plot_idx-1].plot(X[int(N/2), :], P_half, label=f"Time $t={t}$")
        ax3[plot_idx-1].set_ylabel(r"Pressure $P$")
        ax3[plot_idx-1].legend()

        # Increment plot index
        plot_idx += 1
    
print(f"Final value of sigma = {sigma[-1]:.2e}")

# Fix pressure plot
ax3[-1].set_xlabel(r"Coordinate $x$")
ax3[0].set_title(r"Pressure along $x$ axis at $y=M/2$")


fig4, ax4 = plt.subplots(nrows=1, ncols=2, figsize=(12, 9))
ax4[0].plot(t_plot, pressure_diff)
ax4[0].set_title("Pressure Difference")
ax4[1].plot(t_plot, sigma)
ax4[1].set_title("Surface Tension")

plt.show()

# from IPython import embed;embed()