import json
from pathlib import Path

import numpy as np
from PIL import Image

from terrain.config import TerrainConfig

FORMAT_VERSION = 1


def build_metadata(config: TerrainConfig) -> dict:
    """Export of generated terrain data to .npy, png and terrain.json.
    # przetestuje import tez do godot
    """
    return {
        "format_version": FORMAT_VERSION,
        "resolution": config.size,
        "world": {
            "size": config.world_size,
            "height_scale": config.height_scale,
            "sample_spacing": config.world_size / (config.size - 1),
            "units": "world units",
        },
        "seed": config.seed,
        "generator": "fbm_perlin",
        "config": config.to_dict(),
        "coordinate_system": {
            "array_index": "heightmap[row, col]",
            "x": "col * sample_spacing",
            "y": "row * sample_spacing",
            "height": "heightmap value * height_scale",
            "origin": "sample [0, 0] (top-left of the image) is at x=0, y=0",
            "godot_mapping": {"x": "x", "y": "height", "z": "y"},
        },
        "files": {
            "heightmap.npy": {
                "dtype": "float32",
                "shape": [config.size, config.size],
                "value_range": [0.0, 1.0],
                "description": "normalized heights, source of truth",
            },
            "heightmap.png": {
                "dtype": "uint8",
                "shape": [config.size, config.size],
                "description": "8-bit preview of heightmap.npy (value = round(h * 255))",
            },
        },
    }


def export_terrain(heightmap: np.ndarray, config: TerrainConfig, output_dir) -> dict:
    """Write heightmap.npy, heightmap.png and terrain.json to output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "heightmap.npy", heightmap)

    preview = np.round(heightmap * 255).astype(np.uint8)
    Image.fromarray(preview).save(output_dir / "heightmap.png")

    metadata = build_metadata(config)
    (output_dir / "terrain.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata