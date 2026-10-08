"""
Core data structure for a generated city.

A CityGrid is a stack of 2D numpy arrays, one per "channel", all the same
(size x size) shape -- effectively a multi-band image. This is the shared
data format that the terrain model, the building model, and the Blender
importer all read and write, so every part of the pipeline can be built and
tested independently as long as it agrees on this shape.

Channels:
    terrain_height  : float32, arbitrary units, ground elevation
    is_water        : bool, True where the cell is water (river/harbor/lake)
    building_height : float32, 0.0 where there is no building
    land_use        : uint8, one of the LAND_* constants below
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LAND_EMPTY = 0
LAND_BUILDING = 1
LAND_ROAD = 2
LAND_GREEN = 3

LAND_USE_NAMES = {
    LAND_EMPTY: "empty",
    LAND_BUILDING: "building",
    LAND_ROAD: "road",
    LAND_GREEN: "green",
}


@dataclass
class CityGrid:
    size: int
    terrain_height: np.ndarray
    is_water: np.ndarray
    building_height: np.ndarray
    land_use: np.ndarray

    def __post_init__(self) -> None:
        shape = (self.size, self.size)
        for name in ("terrain_height", "is_water", "building_height", "land_use"):
            arr = getattr(self, name)
            if arr.shape != shape:
                raise ValueError(
                    f"{name} has shape {arr.shape}, expected {shape}"
                )

    @classmethod
    def empty(cls, size: int) -> "CityGrid":
        return cls(
            size=size,
            terrain_height=np.zeros((size, size), dtype=np.float32),
            is_water=np.zeros((size, size), dtype=bool),
            building_height=np.zeros((size, size), dtype=np.float32),
            land_use=np.zeros((size, size), dtype=np.uint8),
        )

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            size=np.array(self.size),
            terrain_height=self.terrain_height,
            is_water=self.is_water,
            building_height=self.building_height,
            land_use=self.land_use,
        )

    @classmethod
    def load(cls, path: str) -> "CityGrid":
        data = np.load(path)
        return cls(
            size=int(data["size"]),
            terrain_height=data["terrain_height"],
            is_water=data["is_water"],
            building_height=data["building_height"],
            land_use=data["land_use"],
        )

    def stats(self) -> dict:
        buildable = self.land_use == LAND_BUILDING
        return {
            "size": self.size,
            "water_pct": float(self.is_water.mean()) * 100,
            "building_pct": float(buildable.mean()) * 100,
            "green_pct": float((self.land_use == LAND_GREEN).mean()) * 100,
            "road_pct": float((self.land_use == LAND_ROAD).mean()) * 100,
            "max_building_height": float(self.building_height.max()),
            "mean_building_height": float(self.building_height[buildable].mean())
            if buildable.any()
            else 0.0,
        }
