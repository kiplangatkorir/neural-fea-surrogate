import numpy as np
import matplotlib.pyplot as plt

from fea.plate_rm import solve_plate_rm

# --------------------------------------------------
# Define load function
# --------------------------------------------------
def uniform_load(x, y):
    return np.ones_like(x)

# --------------------------------------------------
# Solve
# --------------------------------------------------
w = solve_plate_rm(
    q_func=uniform_load,
    E=200e9,        # Pa
    nu=0.3,
    h=0.01,         # m
    bc_type="clamped",
    n=64,
)

# --------------------------------------------------
# Visualize
# --------------------------------------------------
plt.imshow(w, origin="lower", cmap="viridis")
plt.colorbar(label="Displacement w")
plt.title("Reissner–Mindlin Plate: Uniform Load")
plt.show()
