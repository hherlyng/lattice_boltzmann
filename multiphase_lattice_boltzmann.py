import numpy             as np
import matplotlib.pyplot as plt

#------------------------------------#
#-----------INITIALIZATION-----------#
#------------------------------------#
# Constants
N = 150
omega = 1.8
A = 0.05
beta = 0.5
Ndir = 9
n = 1
k   = 2*np.pi*n/N # Frequency
idx = np.arange(Ndir)
num_timesteps = 100

t_plot   = np.arange(num_timesteps)
sigma    = np.array([0]*num_timesteps, dtype=np.float64)
pressure_diff = np.copy(sigma)
h_quarter = np.copy(sigma)

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

sine_flag = 1

if sine_flag:
    # Red fluid at bottom, blue fluid on top
    Ri[np.where(Y >= N/2 + 10*int(N/150)*np.sin(k*X))] = 1
    Bi[np.where(Y < N/2 + 10*int(N/150)*np.sin(k*X))] = 1

else:
    # Red fluid circle in center, blue fluid outside
    Ri[circle,  :] = 1
    Bi[outside, :] = 1

# Total population
Ni = Ri+Bi

Ni_eq      = np.zeros_like(Ni)
color_term = np.zeros_like(Ni)
cos_phi    = np.zeros_like(Ni)

# Create plot for visualizing the distribtuion at different times
fig2, ax2 = plt.subplots(nrows=2, ncols=2, figsize=(12, 9), layout='constrained')
ax2 = ax2.flatten()
ax2[0].set_title("Initial Condition", fontsize=35)
RGB = np.zeros((N, N, 3))
R_summed = Ri.sum(axis=2)
B_summed = Bi.sum(axis=2)
RGB[:, :, 0] = R_summed
RGB[:, :, 2] = B_summed
im = ax2[0].imshow(RGB)
plot_times = [1, int(num_timesteps*1/3), int(num_timesteps*2/3), num_timesteps]
plot_idx = 0
[ax2[i].tick_params(axis='both', labelsize=17) for i in range(len(plot_times))]
[ax2[i].set_xticks([0, int(N/5), int(2*N/5), int(3*N/5), int(4*N/5), N-1]) for i in range(len(plot_times))]
[ax2[i].set_yticks([0, int(N/5), int(2*N/5), int(3*N/5), int(4*N/5), N-1]) for i in range(len(plot_times))]
[ax2[i].invert_yaxis() for i in range(len(plot_times))]

# Create plot for visualizing the centerline pressure at different times
fig3, ax3 = plt.subplots(nrows=4, ncols=1, figsize=(12, 9), sharex=True, layout='constrained')


#-----------------------------------#
#---------UTILITY FUNCTIONS---------#
#-----------------------------------#

def calc_color_gradient(Ri, Bi):
    # Color gradient components
    fx = np.zeros((N, N), dtype=np.float64)
    fy = np.zeros((N, N), dtype=np.float64)

    sum2_x = np.copy(Ri)
    sum2_y = np.copy(Bi)
    # Calculate sum with j as outer, i as inner
    for j in idx:
        for i, cx, cy in zip(idx, ex, ey):
            R_streamed = np.roll(Ri[:, :, j], shift=(cx, cy), axis=(1, 0))
            B_streamed = np.roll(Bi[:, :, j], shift=(cx, cy), axis=(1, 0))
            
            sum2_x[:, :, j] += cx * (R_streamed - B_streamed)
            sum2_y[:, :, j] += cy * (R_streamed - B_streamed)
        
    fx2 = sum2_x.sum(axis=2)
    fy2 = sum2_y.sum(axis=2)

    for i, cx, cy in zip(idx, ex, ey):

        R_temp = np.copy(Ri)
        B_temp = np.copy(Bi)

        # Red fluid at x + ci
        R_temp = np.roll(R_temp, shift=(cx, cy), axis=(1, 0))

        # Blue fluid at x + ci
        B_temp = np.roll(B_temp, shift=(cx, cy), axis=(1, 0))

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

def find_interface(Ri, Bi, quarter=False):
    
    if not quarter:
        interface_y = np.zeros((N, 1))
        for i in range(N):
            red_sum  = Ri[:, i, :].sum(axis=1)
            blue_sum = Bi[:, i, :].sum(axis=1)
            diff = red_sum - blue_sum
            int_idx = 0
            
            for j in range(len(diff)):
                if diff[j] < 0 and diff[j+1] > 0:
                    int_idx = j
                    break
            interface_y[i] = Y[int_idx, i]
        
        return interface_y

    else:
        i = int(N/4)
        red_sum  = Ri[:, i, :].sum(axis=1)
        blue_sum = Bi[:, i, :].sum(axis=1)
        diff = -(red_sum - blue_sum)
        int_idx = 0
        for j in range(1, len(diff)):
            if diff[j-1] < 0 and diff[j] > 0:
                int_idx = j
                
                return Y[int_idx, i]
                
        

#-----------------------------------#
#--------SIMULATION TIMELOOP--------#
#-----------------------------------#

for t in range(1, num_timesteps+1):
    print(f"t = {t}")

    # Calculate fluid variables
    rho = np.sum(Ni, axis=2) # Density
    ux  = np.sum(Ni*ex, axis=2) / rho # x velocity
    uy  = np.sum(Ni*ey, axis=2) / rho # y velocity
    # if t == 0:
    #     uy = 0.5e-3*np.ones_like(uy)

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
    
    # Antidiffusive recoloring step
    Ri, Bi = antidiffusive_recoloring(Ni_prime=Ni, rho=rho, Ri=Ri, Bi=Bi, cos_phi=cos_phi)
    
    # Propagation step
    for i, cx, cy in zip(idx, ex, ey):
        # Red fluid
        Ri[:, :, i] = np.roll(Ri[:, :, i], shift=(cx, cy), axis=(1, 0))
        
        # Blue fluid
        Bi[:, :, i] = np.roll(Bi[:, :, i], shift=(cx, cy), axis=(1, 0))
        
    Ni = Ri+Bi
    
    
    if sine_flag:
        # Store interface at x = N/4
        h_quarter[t-1] = find_interface(Ri, Bi, quarter=True)


    else:
        # Calculate pressure difference over the droplet
        P = rho/3
        idx_center_x  = 70
        idx_center_y  = int(N/2)
        idx_outside_x = 50
        idx_outside_y = int(N/2)
        P_center  = P[idx_center_y, idx_center_x]
        P_outside = P[idx_outside_y, idx_outside_x]
        delta_P = P_center - P_outside
        dist = np.sqrt((X[idx_outside_y, idx_outside_x] - X[idx_center_y, idx_center_x])**2 + (Y[idx_outside_y, idx_outside_x] - Y[idx_center_y, idx_center_x])**2)
        sigma[t-1] = delta_P*dist
        pressure_diff[t-1] = delta_P
        if np.isclose(t, 25000):
            from IPython import embed;embed()

    # Visualize
    if t in plot_times:
        if t != 0:
            # Plot the particle distribution
            ax2[plot_idx].set_title(rf"Time $t = {t}$", fontsize=35)

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

    
        # If sine wave case, plot the interface
        if sine_flag:
            h = find_interface(Ri, Bi, quarter=False)
            ax3[plot_idx].plot(X[0, :], h.flatten(), label=rf"Time $t={t}$")
            ax3[plot_idx].legend(fontsize=20)

        # If circle case, plot the centerline pressure
        else:
            # Plot the centerline pressure
            P_half = P[int(N/2), :]
            ax3[plot_idx].plot(X[int(N/2), :], P_half, label=rf"Time $t={t}$")
            ax3[plot_idx].set_ylabel(r"Pressure $P$", fontsize=25)
            ax3[plot_idx].legend(fontsize=20)

        # Increment plot index
        plot_idx += 1

if sine_flag:
    ax3[1].set_ylabel(r"Interface position $h$", fontsize=25)
    ax3[-1].set_xlabel(r"Coordinate $x$", fontsize=25)
    ax3[-1].tick_params(axis='x', labelsize=17)
    [ax3[i].tick_params(axis='y', labelsize=17) for i in range(len(plot_times))]
    [ax3[i].invert_xaxis() for i in range(len(plot_times))]
    #[ax3[i].invert_yaxis() for i in range(len(plot_times))]
    # Plot the interface position at x = N/4
    fig4, ax4 = plt.subplots(figsize=(12, 9))
    ax4.plot(t_plot, h_quarter)
    ax4.set_title(r"Interface $h$ at $x=N/4$", fontsize=35)
    ax4.set_xlabel(r"Time $t$", fontsize=25)
    ax4.set_ylabel(r"Vertical position $y$", fontsize=25)
    #ax4.invert_yaxis()
    ax4.tick_params(axis='both', labelsize=17)

else:
    # Print final value of the surface tension
    print(f"Final value of sigma = {sigma[-1]:.2e}")
    
    # Fix pressure plot
    ax3[-1].set_xlabel(r"Coordinate $x$", fontsize=25)
    ax3[-1].tick_params(axis='x', labelsize=17)
    [ax3[i].tick_params(axis='y', labelsize=17) for i in range(len(plot_times))]
    [ax3[i].invert_xaxis() for i in range(len(plot_times))]
    #[ax3[i].invert_yaxis() for i in range(len(plot_times))]
    ax3[0].set_title(r"Pressure along $x$ axis at $y=M/2$", fontsize=35)

    # Plot pressure difference and surface tension
    fig4, ax4 = plt.subplots(nrows=2, ncols=1, figsize=(12, 9), sharex=True)
    ax4[0].plot(t_plot, pressure_diff)
    ax4[0].set_title("Pressure Difference", fontsize=35)
    ax4[1].plot(t_plot, sigma)
    ax4[1].set_title("Surface Tension", fontsize=35)
    ax4[1].set_xlabel(r"Time $t$", fontsize=25)

plt.show()