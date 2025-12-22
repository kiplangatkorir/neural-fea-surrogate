import numpy as np

def uniform_load(n, magnitude=1.0):
    """Uniform pressure over the plate."""
    return np.full((n, n), magnitude, dtype=np.float32)

def point_load(n, magnitude=1.0, sigma=0.02):
    """
    Gaussian-smoothed point load.
    Centered at plate center by default.
    """
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    cx, cy = 0.5, 0.5
    load = magnitude * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
    return load.astype(np.float32)

def line_load(n, magnitude=1.0, axis='x', pos=0.5, sigma=0.02):
    """
    Gaussian-smoothed line load along x=pos or y=pos.
    """
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    if axis == 'x':
        load = magnitude * np.exp(-(xx - pos)**2 / (2*sigma**2))
    elif axis == 'y':
        load = magnitude * np.exp(-(yy - pos)**2 / (2*sigma**2))
    else:
        raise ValueError("axis must be 'x' or 'y'")
    return load.astype(np.float32)
