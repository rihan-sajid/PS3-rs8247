import numpy as np
import matplotlib.pyplot as plt

# Parameters
w = 5.0
F_m = 1.0
w_F = 0.1
t_start, t_end = 0, 100
dt = 0.05 # Time step
t = np.arange(t_start, t_end + dt, dt)
N = len(t)

# Exact Solution
x_exact = (1 / (w**2 - w_F**2)) * (np.cos(w_F * t) - np.cos(w * t))
v_exact = (1 / (w**2 - w_F**2)) * (-w_F * np.sin(w_F * t) + w * np.sin(w * t))

# Initialization
x_fe, v_fe = np.zeros(N), np.zeros(N)
x_se, v_se = np.zeros(N), np.zeros(N)
x_rk, v_rk = np.zeros(N), np.zeros(N)

# Force function
def force(t_val): return F_m * np.cos(w_F * t_val)

# Integration Loop
for i in range(N - 1):
    # Forward Euler
    x_fe[i+1] = x_fe[i] + dt * v_fe[i]
    v_fe[i+1] = v_fe[i] + dt * (force(t[i]) - w**2 * x_fe[i])
    
    # Symplectic Euler
    v_se[i+1] = v_se[i] + dt * (force(t[i]) - w**2 * x_se[i])
    x_se[i+1] = x_se[i] + dt * v_se[i+1] # Uses updated velocity
    
    # RK4
    def f(t_val, x_val, v_val):
        return v_val, force(t_val) - w**2 * x_val
        
    k1_x, k1_v = f(t[i], x_rk[i], v_rk[i])
    k2_x, k2_v = f(t[i] + dt/2, x_rk[i] + dt/2 * k1_x, v_rk[i] + dt/2 * k1_v)
    k3_x, k3_v = f(t[i] + dt/2, x_rk[i] + dt/2 * k2_x, v_rk[i] + dt/2 * k2_v)
    k4_x, k4_v = f(t[i] + dt, x_rk[i] + dt * k3_x, v_rk[i] + dt * k3_v)
    
    x_rk[i+1] = x_rk[i] + (dt/6) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
    v_rk[i+1] = v_rk[i] + (dt/6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

# L2 Norm of State Errors at each time t: sqrt((x_num - x_exact)^2 + (v_num - v_exact)^2)
error_fe = np.sqrt((x_fe - x_exact)**2 + (v_fe - v_exact)**2)
error_se = np.sqrt((x_se - x_exact)**2 + (v_se - v_exact)**2)
error_rk = np.sqrt((x_rk - x_exact)**2 + (v_rk - v_exact)**2)

# Plotting
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# 1. Response x(t) vs t
axs[0].plot(t, x_exact, 'k--', label='Exact', linewidth=2)
axs[0].plot(t, x_se, label='Symplectic Euler', alpha=0.8)
axs[0].plot(t, x_rk, label='RK4', alpha=0.8)
# Excluded Forward Euler from pos plot because it explodes and ruins the scale
axs[0].set_title('Response of the System x(t) vs t')
axs[0].set_xlabel('Time (t)')
axs[0].set_ylabel('Position x(t)')
axs[0].legend()

# 2. Phase Portrait v(t) vs x(t)
axs[1].plot(x_exact, v_exact, 'k--', label='Exact')
axs[1].plot(x_fe, v_fe, label='Forward Euler', alpha=0.5)
axs[1].plot(x_se, v_se, label='Symplectic Euler', alpha=0.8)
axs[1].set_title('Phase Portrait v(t) vs x(t)')
axs[1].set_xlabel('Position x(t)')
axs[1].set_ylabel('Velocity v(t)')
axs[1].legend()

# 3. Error Plot (semilog-y)
axs[2].semilogy(t, error_fe, label='Forward Euler')
axs[2].semilogy(t, error_se, label='Symplectic Euler')
axs[2].semilogy(t, error_rk, label='RK4')
axs[2].set_title('L2 Error ||e(t)|| vs t')
axs[2].set_xlabel('Time (t)')
axs[2].set_ylabel('Error Norm')
axs[2].legend()

plt.tight_layout()
plt.show()