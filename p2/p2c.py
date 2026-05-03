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
    
    
    

# Analytical Solution (For Comparison)
C = F_m / (omega**2 - omega_F**2)
x_exact = C * (np.cos(omega_F * t) - np.cos(omega * t))

# Plot the Results 
plt.figure(figsize=(12, 6))

# Plot Numerical
plt.plot(t, x, label=f'Forward Euler (dt={dt})', color='red', alpha=0.8)

# Plot Exact
plt.plot(t, x_exact, label='Exact Analytical', color='black', linestyle='dashed')

plt.title('Harmonically Forced Undamped Oscillator')
plt.xlabel('Time (t)')
plt.ylabel('Displacement x(t)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()