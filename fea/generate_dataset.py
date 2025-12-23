import numpy as np
import h5py
from fea.plate_rm import solve_plate_scalar
from fea.loads import uniform_load, point_load, line_load
from fea.bc_masks import clamped_mask, simply_supported_mask, mixed_mask

# ---------------- Parameters ----------------
N_SAMPLES = 2000       # Number of FEA simulations
GRID = 64              # Grid resolution
OUTPUT_FILE = "data/raw/plate_data.h5"

# ---------------- Sampling options ----------------
bc_options = {
    "clamped": clamped_mask,
    "simply_supported": simply_supported_mask,
    "mixed": mixed_mask
}

load_options = [uniform_load, point_load, line_load]

# Material stiffness range (bending stiffness D)
D_min, D_max = 50.0, 200.0

# ---------------- Dataset generation ----------------
with h5py.File(OUTPUT_FILE, "w") as f:
    for i in range(N_SAMPLES):
        # Random load
        load_fn = np.random.choice(load_options)
        magnitude = np.random.uniform(0.5, 2.0)
        if load_fn == line_load:
            axis = np.random.choice(['x', 'y'])
            pos = np.random.uniform(0.25, 0.75)
            q = load_fn(GRID, magnitude=magnitude, axis=axis, pos=pos)
        else:
            q = load_fn(GRID, magnitude=magnitude)

        # Random BC
        bc_name = np.random.choice(list(bc_options.keys()))
        bc_mask = bc_options[bc_name](GRID)

        # Random material stiffness
        D = np.random.uniform(D_min, D_max)

        # Solve plate
        w = solve_plate_scalar(lambda x, y: q, D=D, bc_type='clamped', n=GRID)

        # Save to HDF5
        grp = f.create_group(f"{i:05d}")
        grp.create_dataset("load", data=q)
        grp.create_dataset("bc_mask", data=bc_mask)
        grp.create_dataset("disp", data=w)
        grp.attrs["D"] = D
        grp.attrs["bc_type"] = bc_name
        grp.attrs["magnitude"] = magnitude
