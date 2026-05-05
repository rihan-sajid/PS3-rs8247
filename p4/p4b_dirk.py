# dirk2_solver.py
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt

# Rate constants
k1_f = 2e3; k1_b = 3e12; k2_f = 2e1
gamma = 1.0 - 1.0 / np.sqrt(2.0)

def f_sys(t, y):
    X_N, X_O, X_N2, X_O2, X_NO = y
    r1 = k1_f * X_N2 * X_O - k1_b * X_NO * X_N
    r2 = k2_f * X_N * X_O2
    return np.array([r1 - r2, -r1 + r2, -r1, -r2, r1 + r2])

def dirk2_step(t_n, y_n, dt):
    """2-stage L-stable Diagonally Implicit Runge-Kutta step."""
    
    # Stage 1: Solve for Y1
    def res1(Y1):
        return Y1 - y_n - dt * gamma * f_sys(t_n + gamma*dt, Y1)
    
    sol1 = root(res1, y_n)
    if not sol1.success: return None
    Y1 = sol1.x
    
    # Stage 2: Solve for Y2 (which is also y_{n+1})
    def res2(Y2):
        return Y2 - y_n - dt * (1-gamma) * f_sys(t_n + gamma*dt, Y1) - dt * gamma * f_sys(t_n + dt, Y2)
    
    sol2 = root(res2, Y1)
    if not sol2.success: return None
    return sol2.x

if __name__ == "__main__":
    y = np.array([0.01, 0.01, 0.75, 0.23, 0.00])
    t = 0.0
    t_end = 40.0
    dt = 1e-12 # Start tiny due to extreme initial transient
    
    t_vals = [t]; y_vals = [y]
    
    print("Running Adaptive DIRK2 to t = 40s...")
    while t < t_end:
        # Cap dt so we land exactly on t_end
        dt = min(dt, t_end - t)
        
        y_next = dirk2_step(t, y, dt)
        
        if y_next is None: # Newton solver failed
            dt *= 0.5
            continue
            
        # 10% Variation Rule for Adaptive Stepping
        # Add small epsilon to denominator to prevent division by zero
        variation = np.max(np.abs((y_next - y) / (np.abs(y) + 1e-12)))
        
        if variation > 0.10:
            # Reject step
            dt *= 0.5
        else:
            # Accept step
            t += dt
            y = y_next
            t_vals.append(t)
            y_vals.append(y)
            
            # Increase step size if variation is very small
            if variation < 0.02:
                dt *= 1.5

    y_vals = np.array(y_vals)
    print(f"DIRK2 reached {t_end}s in {len(t_vals)} steps.")
    
    plt.figure()
    for i, label in enumerate(['N', 'O', 'N2', 'O2', 'NO']):
        plt.plot(t_vals, y_vals[:, i], label=label)
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('Time (s)'); plt.ylabel('Mole Fraction')
    plt.title('Adaptive DIRK2')
    plt.legend(); plt.grid(True)
    plt.savefig('p4b_dirk2.png')
    plt.show()