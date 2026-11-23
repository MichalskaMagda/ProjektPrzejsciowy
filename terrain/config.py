import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass(frozen=True)
class TerrainConfig:
    """ Terrain generator configuration module. Config relies on seed and is reproducable with the same one. 
    (proponuje parametry size (size x size probek), seed, scale, 
    octaves (liczba warstw szumu ktore sie sumuja), 
    persistence, lacunarity (mnozniki czestotliwosci i amplitudy kolejnych warstw wedlug perina),
    world_size, height_scale (jednostki w silniku wizualizacji skrypt przetworzy))
    https://thebookofshaders.com/13/
    """
    size: int = 256
    seed: int = 12345
    scale: float = 70.0
    octaves: int = 5
    persistence: float = 0.5
    lacunarity: float = 2.0
    world_size: float = 1000.0
    height_scale: float = 100.0

    def __post_init__(self):
        if self.size < 2:
            raise ValueError("size must be >= 2")
        if self.scale <= 0:
            raise ValueError("scale must be > 0")
        if self.octaves < 1:
            raise ValueError("octaves must be >= 1")
        if not 0 < self.persistence <= 1:
            raise ValueError("persistence must be in (0, 1]")
        if self.lacunarity < 1:
            raise ValueError("lacunarity must be >= 1")
        if self.world_size <= 0 or self.height_scale <= 0:
            raise ValueError("world_size and height_scale must be > 0")

    @classmethod
    def from_json(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    def to_dict(self):
        return asdict(self)