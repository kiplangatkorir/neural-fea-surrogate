import numpy as np
from fea.plate_rm import solve_plate_rm

# Load function: uniform
def q_uniform(x, y):
    return np.ones_like(x) * 1.0  # Unit load

# Material parameters
E = 150.0
nu = 0.3
h = 0.02
n = 64

w = solve_plate_rm(q_uniform, E, nu, h, bc_type="clamped", n=n)

# Center deflection
center_deflection = w[n//2, n//2]
print("Center deflection:", center_deflection)

# Check symmetry along axes
sym_x = np.allclose(w, w[::-1, :], atol=1e-6)
sym_y = np.allclose(w, w[:, ::-1], atol=1e-6)
print("Symmetry X:", sym_x, "Symmetry Y:", sym_y)
