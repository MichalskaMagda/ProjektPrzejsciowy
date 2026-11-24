from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

from terrain.config import TerrainConfig
from terrain.generator import generate_heightmap
from terrain.validation import validate_heightmap

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


def check_validation_rejects_bad_data(config):
    good = generate_heightmap(config)
    with_nan = good.copy()
    with_nan[0, 0] = np.nan
    cases = {
        "NaN": with_nan,
        "out of range": good * 2.0,
        "wrong dtype": good.astype(np.float64),
        "wrong shape": good[:-1, :],
    }
    for name, bad in cases.items():
        try:
            validate_heightmap(bad, config)
        except ValueError as error:
            print(f"rejected {name}: {error}")
        else:
            raise AssertionError(f"validation accepted bad data: {name}")


check_validation_rejects_bad_data(base)
print()

fig, axes = plt.subplots(len(sweeps), 3, figsize=(9, 15))
print(f"{'param':<12}{'value':>8}{'mean':>8}{'std':>8}{'roughness':>12}")
for row, (name, values) in enumerate(sweeps.items()):
    for col, value in enumerate(values):
        config = replace(base, **{name: value})
        heightmap = generate_heightmap(config)
        validate_heightmap(heightmap, config)
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