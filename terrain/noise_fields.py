"""aux blocks for procedural height fields (noise layers, transforms, )"""
import numpy as np
from noise import pnoise2


def perlin(size, wavelength, offset):
    """perlin on a (size, size) grid, z ~ [-1, 1]."""
    grid = np.empty((size, size), dtype=np.float32)
    for y in range(size):
        for x in range(size):
            grid[y, x] = pnoise2((x + offset[0]) / wavelength, (y + offset[1]) / wavelength)
    return grid


# from default config
def fractal(size, offset, wavelength=70.0, octaves=5, persistence=0.5, lacunarity=2.0, transform=None):
    """Sum of noise (transformed before) octaves"""
    total = np.zeros((size, size), dtype=np.float32)
    amplitude = 1.0
    for _ in range(octaves):
        layer = perlin(size, wavelength, offset)
        total += amplitude * (transform(layer) if transform else layer)
        amplitude *= persistence
        wavelength /= lacunarity
    return total


def normalize(field):
    low, high = float(field.min()), float(field.max())
    if high - low < 1e-9:
        raise ValueError("field is flat, cannot normalize")
    return ((field - low) / (high - low)).astype(np.float32)


def ridged(layer):
    return (1.0 - np.abs(layer)) ** 2


def billow(layer):
    return np.abs(layer)


def smoothstep(edge0, edge1, x):
    """0-1 in -> transition between edge0 and edge1."""
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)