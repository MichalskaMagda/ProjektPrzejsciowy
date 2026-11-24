import numpy as np

from terrain.config import TerrainConfig


def validate_heightmap(heightmap: np.ndarray, config: TerrainConfig) -> None:
    """raise default valueerros for cli run"""
    expected_shape = (config.size, config.size)
    if heightmap.shape != expected_shape:
        raise ValueError(f"expected shape {expected_shape}, got {heightmap.shape}")
    if heightmap.dtype != np.float32:
        raise ValueError(f"expected dtype float32, got {heightmap.dtype}")
    if not np.isfinite(heightmap).all():
        raise ValueError("heightmap contains NaN or inf")
    if heightmap.min() < 0.0 or heightmap.max() > 1.0:
        raise ValueError("heightmap values must be in [0, 1]")