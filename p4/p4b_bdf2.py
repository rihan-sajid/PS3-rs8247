# bdf2_solver.py
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt

# Rate constants
k1_f = 2e3; k1_b = 3e12; k2_f = 2e1

def f_sys(t, y):
    X_N, X_O, X_N2, X_O2, X_NO = y
    r1 = k1_f * X_N2 * X_O - k1_b * X_NO * X_N
    r2 = k2_f * X_N * X_O2
    return np.array([r1 - r2, -r1 + r2, -r1, -r2, r1 + r2])

def bdf1_step(t_n, y_n, dt):
    """Backward Euler (BDF1) used only to bootstrap the first step."""
    def res(y_next):
        return y_next - y_n - dt * f_sys(t_n + dt, y_next)
    sol = root(res, y_n)
    return sol.x if sol.success else None

def bdf2_step(t_next, y_n, y_prev, dt_n, dt_prev):
    """
    BDF2 step accounting for variable time steps.
    If dt_n == dt_prev, this simplifies to standard BDF2.
    """
    rho = dt_n / dt_prev
    c1 = (1 + rho)**2 / (1 + 2*rho)
    c2 = -rho**2 / (1 + 2*rho)
    c3 = dt_n * (1 + rho) / (1 + 2*rho)
    
    def res(y_next):
        return y_next - c1*y_n - c2*y_prev - c3*f_sys(t_next, y_next)
    
    sol = root(res, y_n)
    return sol.x if sol.success else None

if __name__ == "__main__":
    y0 = np.array([0.01, 0.01, 0.75, 0.23, 0.00])
    t = 0.0; t_end = 40.0
    dt = 1e-12 
    
    t_vals = [t]; y_vals = [y0]
    
    # Bootstrap the first step using BDF1
    y1 = bdf1_step(t, y0, dt)
    t += dt
    t_vals.append(t); y_vals.append(y1)
    
    dt_prev = dt
    
    print("Running Adaptive BDF2 to t = 40s...")
    while t < t_end:
        dt = min(dt, t_end - t)
        
        y_prev = y_vals[-2]
        y_n = y_vals[-1]
        
        y_next = bdf2_step(t + dt, y_n, y_prev, dt, dt_prev)
        
        if y_next is None:
            dt *= 0.5
            continue
            
        variation = np.max(np.abs((y_next - y_n) / (np.abs(y_n) + 1e-12)))
        
        if variation > 0.10:
            dt *= 0.5 # Reject and shrink
        else:
            t += dt
            y_vals.append(y_next)
            t_vals.append(t)
            
            dt_prev = dt # Store accepted step size for next BDF2 interpolation
            
            if variation < 0.02:
                dt *= 1.5 # Expand
                
    y_vals = np.array(y_vals)
    print(f"BDF2 reached {t_end}s in {len(t_vals)} steps.")
    
    plt.figure()
    for i, label in enumerate(['N', 'O', 'N2', 'O2', 'NO']):
        plt.plot(t_vals, y_vals[:, i], label=label)
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('Time (s)'); plt.ylabel('Mole Fraction')
    plt.title('Adaptive BDF2')
    plt.legend(); plt.grid(True)
    plt.savefig('p4b_bdf2.png')
    plt.show()