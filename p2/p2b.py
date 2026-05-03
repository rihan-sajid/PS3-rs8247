import numpy as np
import matplotlib.pyplot as plt

# Parameters
omega = 5.0
F_m = 1.0        
omega_F = 0.1
t_max = 100.0
dt = 0.005        

t = np.arange(0, t_max + dt, dt)
n_steps = len(t)

# Arrays to store numerical results
x = np.zeros(n_steps)
v = np.zeros(n_steps)

x[0] = 0.0
v[0] = 0.0

# Main loop
for i in range(n_steps - 1):
    # Calculate acceleration based on the current state: a = (F/m)*cos(wF*t) - w^2*x
    a_i = F_m * np.cos(omega_F * t[i]) - (omega**2) * x[i]
    
    # Update position and velocity using Forward Euler formulas
    v[i+1] = v[i] + dt * a_i
    x[i+1] = x[i] + dt * v[i+1]

# Analytical Solution (For Comparison)
C = F_m / (omega**2 - omega_F**2)
x_exact = C * (np.cos(omega_F * t) - np.cos(omega * t))
v_exact = C * (-omega_F * np.sin(omega_F * t) + omega * np.sin(omega * t))

error_fe = np.sqrt((x - x_exact)**2 + (v - v_exact)**2)

# Plot the Results 
figs, axes = plt.subplots(2, 1, figsize=(12, 10))

# Plot Numerical
axes[0].plot(t, x, label=f'Forward Euler (dt={dt})', color='red', alpha=0.8)

# Plot Exact
axes[0].plot(t, x_exact, label='Exact Analytical', color='black', linestyle='dashed')

axes[0].set_title('Harmonically Forced Undamped Oscillator')
axes[0].set_xlabel('Time (t)')
axes[0].set_ylabel('Displacement x(t)')
axes[0].legend()
axes[0].grid(True)

# Plot Error
axes[1].semilogy(t, error_fe, label='Error (Forward Euler)', color='blue', alpha=0.8)
axes[1].set_title('Error in Forward Euler Method')
axes[1].set_xlabel('Time (t)')
axes[1].set_ylabel('Error')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()

# Show the plot
plt.show()