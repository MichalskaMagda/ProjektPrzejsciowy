from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

from terrain.config import TerrainConfig
from terrain.generator import generate_heightmap

base = TerrainConfig.from_json("configs/default.json")

sweeps = {
    "scale": [30.0, 70.0, 150.0],
    "octaves": [1, 3, 8],
    "persistence": [0.3, 0.5, 0.8],
    "lacunarity": [1.5, 2.0, 3.0],
    "seed": [1, 12345, 999],
}


def roughness(heightmap):
    gy, gx = np.gradient(heightmap)
    return float(np.hypot(gx, gy).mean())


fig, axes = plt.subplots(len(sweeps), 3, figsize=(9, 15))
print(f"{'param':<12}{'value':>8}{'mean':>8}{'std':>8}{'roughness':>12}")
for row, (name, values) in enumerate(sweeps.items()):
    for col, value in enumerate(values):
        heightmap = generate_heightmap(replace(base, **{name: value}))
        print(
            f"{name:<12}{value:>8}{heightmap.mean():>8.3f}"
            f"{heightmap.std():>8.3f}{roughness(heightmap):>12.5f}"
        )
        ax = axes[row, col]
        ax.imshow(heightmap, cmap="terrain", vmin=0, vmax=1)
        ax.set_title(f"{name}={value}", fontsize=9)
        ax.axis("off")
plt.tight_layout()
plt.show()