import numpy as np
from noise import pnoise2

from terrain.config import TerrainConfig

_OFFSET_RANGE = 10000.0 

def generate_heightmap(config: TerrainConfig) -> np.ndarray:
    """ returns a normalized [0, 1] heightmap of shape (size, size) with values in [0, 1];
    gets its parameters from config (terrain.config.TerrainConfig). Row index is the Y axis, column index is the X axis.
    
    # obecnie offset jest losowy, ale mozna go ustalic zeby generowac powtarzalne mapy przy tym samym seedzie
    """
    rng = np.random.default_rng(config.seed)
    offset_x, offset_y = rng.uniform(0.0, _OFFSET_RANGE, size=2)

    heightmap = np.zeros((config.size, config.size), dtype=np.float32)
    for y in range(config.size):
        for x in range(config.size):
            heightmap[y, x] = pnoise2(
                (x + offset_x) / config.scale,
                (y + offset_y) / config.scale,
                octaves=config.octaves,
                persistence=config.persistence,
                lacunarity=config.lacunarity,
            )
    return _normalize(heightmap)


def _normalize(heightmap: np.ndarray) -> np.ndarray:
    low = float(heightmap.min())
    high = float(heightmap.max())
    if high - low < 1e-9:
        raise ValueError("heightmap is flat")
    return ((heightmap - low) / (high - low)).astype(np.float32)