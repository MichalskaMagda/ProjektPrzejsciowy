import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource
from noise import pnoise2
from scipy.ndimage import map_coordinates

from terrain.config import TerrainConfig

config = TerrainConfig.from_json("configs/default.json")
spacing = config.world_size / (config.size - 1)
offsets = np.random.default_rng(config.seed).uniform(0.0, 10000.0, size=(3, 2))


def perlin(wavelength, offset):
    grid = np.empty((config.size, config.size), dtype=np.float32)
    for y in range(config.size):
        for x in range(config.size):
            grid[y, x] = pnoise2(
                (x + offset[0]) / wavelength, (y + offset[1]) / wavelength
            )
    return grid


def fractal(offset, wavelength=70.0, octaves=5, persistence=0.5, lacunarity=2.0, transform=None):
    total = np.zeros((config.size, config.size), dtype=np.float32)
    amplitude = 1.0
    for _ in range(octaves):
        layer = perlin(wavelength, offset)
        total += amplitude * (transform(layer) if transform else layer)
        amplitude *= persistence
        wavelength /= lacunarity
    return total


def normalize(field):
    return ((field - field.min()) / (field.max() - field.min())).astype(np.float32)


def billow(layer):
    return np.abs(layer)


def ridged(layer):
    return (1.0 - np.abs(layer)) ** 2


warp_dx = fractal(offsets[1], 120.0, octaves=2)
warp_dy = fractal(offsets[2], 120.0, octaves=2)
rows, cols = np.mgrid[0 : config.size, 0 : config.size]


def warp(field, strength):
    return map_coordinates(
        field, [rows + strength * warp_dy, cols + strength * warp_dx], order=1, mode="reflect"
    )


def slope_stats(field):
    gy, gx = np.gradient(field)
    slope = np.hypot(gx, gy) * config.height_scale / spacing
    return float(slope.mean()), float((slope > 0.5).mean() * 100)


fbm = normalize(fractal(offsets[0]))
ridged_field = normalize(fractal(offsets[0], transform=ridged))
variants = {
    "fBm (baseline)": fbm,
    "fBm ** 2.5": fbm**2.5,
    "billow": normalize(fractal(offsets[0], transform=billow)),
    "ridged": ridged_field,
    "fBm + warp": normalize(warp(fbm, 40.0)),
    "ridged + warp": normalize(warp(ridged_field, 40.0)),
}

light = LightSource(azdeg=315, altdeg=45)
fig, axes = plt.subplots(2, 3, figsize=(13, 9))
print(f"{'variant':<18}{'mean h':>8}{'mean slope':>12}{'steep >0.5 %':>14}")
for ax, (name, field) in zip(axes.flat, variants.items()):
    mean_slope, steep = slope_stats(field)
    print(f"{name:<18}{field.mean():>8.3f}{mean_slope:>12.3f}{steep:>14.1f}")
    shaded = light.shade(
        field, cmap=plt.cm.terrain, vert_exag=config.height_scale, dx=spacing, dy=spacing,
        blend_mode="soft",
    )
    ax.imshow(shaded)
    ax.set_title(name)
    ax.axis("off")
plt.tight_layout()
plt.show()