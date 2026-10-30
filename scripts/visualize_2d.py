#!/usr/bin/env python3
"""
Quick 2D preview of a saved CityGrid -- terrain, water, building height,
and land use, side by side. Use this to sanity-check a generated city in
seconds, before spending time importing it into Blender.

Example:
    python scripts/visualize_2d.py data/city_harbor.npz
    python scripts/visualize_2d.py data/city_harbor.npz --out preview.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from citygen import CityGrid, LAND_BUILDING, LAND_EMPTY, LAND_GREEN, LAND_ROAD

LAND_USE_COLORS = {
    LAND_EMPTY: (0.85, 0.85, 0.8),
    LAND_BUILDING: (0.75, 0.2, 0.2),
    LAND_ROAD: (0.3, 0.3, 0.3),
    LAND_GREEN: (0.25, 0.6, 0.25),
}


def land_use_rgb(land_use: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*land_use.shape, 3), dtype=np.float32)
    for code, color in LAND_USE_COLORS.items():
        rgb[land_use == code] = color
    return rgb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("city_path", type=str)
    parser.add_argument("--out", type=str, default=None, help="save PNG instead of showing")
    args = parser.parse_args()

    city = CityGrid.load(args.city_path)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    terrain_display = np.where(city.is_water, np.nan, city.terrain_height)
    im0 = axes[0].imshow(terrain_display, cmap="terrain")
    axes[0].set_title("Terrain height")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    axes[1].imshow(city.is_water, cmap="Blues")
    axes[1].set_title("Water")

    im2 = axes[2].imshow(city.building_height, cmap="magma")
    axes[2].set_title("Building height")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    axes[3].imshow(land_use_rgb(city.land_use))
    axes[3].set_title("Land use\n(red=building, gray=road, green=park)")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"{args.city_path}  ({city.size}x{city.size})")
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"saved {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()