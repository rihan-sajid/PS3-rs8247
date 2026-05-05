# rk4_solver.py
import numpy as np
import matplotlib.pyplot as plt

# Rate constants
k1_f = 2e3
k1_b = 3e12
k2_f = 2e1

def f_sys(t, y):
    """Evolution equations for the Zeldovich mechanism."""
    X_N, X_O, X_N2, X_O2, X_NO = y
    r1 = k1_f * X_N2 * X_O - k1_b * X_NO * X_N
    r2 = k2_f * X_N * X_O2
    
    return np.array([
        r1 - r2,   # dX_N/dt
        -r1 + r2,  # dX_O/dt
        -r1,       # dX_N2/dt
        -r2,       # dX_O2/dt
        r1 + r2    # dX_NO/dt
    ])

def rk4_step(f, t, y, dt):
    """Standard explicit 4th-order Runge-Kutta step."""
    k1 = f(t, y)
    k2 = f(t + dt/2, y + dt/2 * k1)
    k3 = f(t + dt/2, y + dt/2 * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

if __name__ == "__main__":
    y0 = np.array([0.01, 0.01, 0.75, 0.23, 0.00])
    
    # RK4 requires a tiny dt for stability due to the extreme stiffness
    dt = 1e-11 
    t_end = 1e-9 # Only simulating a fraction of a microsecond
    
    t_vals = [0.0]
    y_vals = [y0]
    
    t = 0.0
    y = y0
    
    print("Running RK4 with fixed dt = 1e-11...")
    while t < t_end:
        y = rk4_step(f_sys, t, y, dt)
        t += dt
        t_vals.append(t)
        y_vals.append(y)
        
    y_vals = np.array(y_vals)
    print("RK4 completed successfully (for tiny timescale).")
    
    plt.plot(t_vals, y_vals[:, 0], label='N')
    plt.plot(t_vals, y_vals[:, 4], label='NO')
    plt.xlabel('Time (s)')
    plt.ylabel('Mole Fraction')
    plt.title('Explicit RK4 (Tiny Timescale)')
    plt.legend()
    plt.savefig('p4b_rk4.png')
    plt.show()