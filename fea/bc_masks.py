import numpy as np

def clamped_mask(n):
    """Mask = 1 at clamped edges, 0 elsewhere."""
    mask = np.zeros((n, n), dtype=np.float32)
    mask[0, :] = 1.0  # left edge
    mask[-1, :] = 1.0 # right edge
    mask[:, 0] = 1.0  # bottom edge
    mask[:, -1] = 1.0 # top edge
    return mask

def simply_supported_mask(n):
    """Mask = 1 at supported edges (e.g., left+right), 0 elsewhere."""
    mask = np.zeros((n, n), dtype=np.float32)
    mask[0, :] = 1.0  # left
    mask[-1, :] = 1.0 # right
    return mask

def mixed_mask(n):
    """Example: left+bottom clamped, right+top free."""
    mask = np.zeros((n, n), dtype=np.float32)
    mask[0, :] = 1.0  # left
    mask[:, 0] = 1.0  # bottom
    return mask
