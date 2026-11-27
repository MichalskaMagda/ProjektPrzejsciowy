import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource

from terrain.config import TerrainConfig
from terrain.noise_fields import fractal, normalize, ridged, smoothstep

config = TerrainConfig.from_json("configs/default.json")
spacing = config.world_size / (config.size - 1)
offsets = np.random.default_rng(config.seed).uniform(0.0, 10000.0, size=(3, 2))
BASE_AMPLITUDE = 0.3

base = normalize(fractal(config.size, offsets[0]))
ridges = normalize(fractal(config.size, offsets[1], wavelength=80.0, transform=ridged))
mask_field = normalize(fractal(config.size, offsets[2], wavelength=160.0, octaves=2))


def mountain_mask(coverage, softness=0.08):
    threshold = np.quantile(mask_field, 1.0 - coverage)
    return smoothstep(threshold - softness, threshold + softness, mask_field)


def build(coverage, mountain_height):
    mask = mountain_mask(coverage)
    return BASE_AMPLITUDE * base + mountain_height * mask * ridges, mask


def slope_map(field):
    gy, gx = np.gradient(field)
    return np.hypot(gx, gy) * config.height_scale / spacing


cases = [(0.15, 0.5), (0.35, 0.5), (0.6, 0.5), (0.35, 0.2), (0.35, 0.5), (0.35, 0.7)]
light = LightSource(azdeg=315, altdeg=45)
fig, axes = plt.subplots(2, 3, figsize=(13, 9))
print(f"{'coverage':>9}{'m_height':>9}{'max h':>7}{'slope in':>10}{'slope out':>11}{'steep %':>9}")
for ax, (coverage, mountain_height) in zip(axes.flat, cases):
    field, mask = build(coverage, mountain_height)
    slope = slope_map(field)
    inside = mask > 0.5
    print(
        f"{coverage:>9}{mountain_height:>9}{field.max():>7.2f}{slope[inside].mean():>10.3f}"
        f"{slope[~inside].mean():>11.3f}{(slope > 0.5).mean() * 100:>9.1f}"
    )
    shaded = light.shade(
        field, cmap=plt.cm.terrain, vmin=0.0, vmax=1.0, vert_exag=config.height_scale,
        dx=spacing, dy=spacing, blend_mode="soft",
    )
    ax.imshow(shaded)
    ax.contour(mask, levels=[0.5], colors="red", linewidths=0.8)
    ax.set_title(f"coverage={coverage}, mountain_height={mountain_height}")
    ax.axis("off")
plt.tight_layout()
plt.show()