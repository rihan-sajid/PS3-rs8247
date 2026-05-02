import numpy as np
import matplotlib.pyplot as plt

# Create complex plane meshgrid
x = np.linspace(-3, 1, 400)
y = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

# Evaluate max root magnitudes
max_r_2nd = np.maximum(np.abs(Z + np.sqrt(Z**2 + 1)), np.abs(Z - np.sqrt(Z**2 + 1)))

max_r_3rd = np.zeros_like(Z, dtype=float)
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        # Roots of: r^3 + (1.5 - 3z)r^2 - 3r + 0.5 = 0
        roots = np.roots([1, 1.5 - 3*Z[i,j], -3, 0.5])
        max_r_3rd[i, j] = np.max(np.abs(roots))

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
levels = np.linspace(1, 10, 20)

c1 = axes[0].contourf(X, Y, max_r_2nd, levels=levels, extend='both', cmap='Reds')
axes[0].set_title('2nd Order: Max |r| Contours')
axes[0].grid(True)
fig.colorbar(c1, ax=axes[0])

c2 = axes[1].contourf(X, Y, max_r_3rd, levels=levels, extend='both', cmap='Reds')
axes[1].set_title('3rd Order: Max |r| Contours')
axes[1].grid(True)
fig.colorbar(c2, ax=axes[1])

plt.show()