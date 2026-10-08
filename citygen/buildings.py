"""
Procedural building placement, conditioned on terrain.

Takes the terrain generator's output and decides where buildings, roads,
and green space go. Conditioning on terrain is what keeps buildings out of
lakes and off cliffs.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .city_grid import LAND_BUILDING, LAND_EMPTY, LAND_GREEN, LAND_ROAD

CITY_ARCHETYPES = ("monocentric", "polycentric", "linear")


# --------------------------------------------------------------------
# Cluster centers -- where the "downtowns" sit
# --------------------------------------------------------------------

def _cluster_centers(archetype: str, size: int, num_clusters: int,
                     rng: np.random.Generator):
    """
    Returns a list of (cx, cy) coordinates -- the centers that buildings
    cluster around. The archetype decides how they're arranged.
    """
    if archetype == "monocentric":
        # one downtown, dead center (this is Week 1's behavior)
        return [(size / 2, size / 2)]

    if archetype == "polycentric":
        # TODO 1: several scattered downtowns.
        # Return a list of num_clusters (cx, cy) tuples at random positions.
        # Keep them away from the very edge -- sample each coordinate
        # between size*0.15 and size*0.85 using rng.uniform(low, high).
        # Use max(num_clusters, 1) so a caller passing 0 still gets a city.
        return [(rng.uniform(size * 0.15, size * 0.85),
                 rng.uniform(size * 0.15, size * 0.85))
                for _ in range(max(num_clusters, 1))]

    if archetype == "linear":
        # GIVEN: centers strung out along a random straight line, which
        # gives the long corridor-shaped cities you see along rivers,
        # coastlines, and highways.
        n = max(num_clusters, 3)
        angle = rng.uniform(0, np.pi)
        centers = []
        for t in np.linspace(0.15, 0.85, n):
            cx = size / 2 + (t - 0.5) * size * np.cos(angle)
            cy = size / 2 + (t - 0.5) * size * np.sin(angle)
            centers.append((cx, cy))
        return centers

    raise ValueError(f"unknown city archetype {archetype!r}, choose from {CITY_ARCHETYPES}")


# --------------------------------------------------------------------
# GIVEN -- road gap mask, no changes needed
# --------------------------------------------------------------------

def _road_keep_mask(size: int, rng: np.random.Generator, continuity: float) -> np.ndarray:
    """
    A smooth random field, thresholded, deciding which candidate road cells
    actually get built.

    Why not just build every road? A perfectly complete grid reads as
    obviously computer-generated. Blurring the noise first (rather than
    using raw per-cell randomness) means the gaps come in connected
    patches -- whole missing stretches of street, like a real city -- not
    scattered single-cell holes.

    Lower continuity = more and larger gaps.
    """
    noise = rng.normal(size=(size, size)).astype(np.float32)
    noise = gaussian_filter(noise, sigma=max(size / 40, 1.0))
    noise -= noise.min()
    peak = noise.max()
    noise = noise / peak if peak > 1e-8 else noise
    threshold = 1.0 - continuity
    return noise > threshold


# --------------------------------------------------------------------
# YOUR CODE -- generate_buildings
# --------------------------------------------------------------------

def generate_buildings(
    terrain_height: np.ndarray,
    is_water: np.ndarray,
    archetype: str = "monocentric",
    num_clusters: int = 1,
    density: float = 0.5,
    cluster_tightness: float = 0.5,
    green_amount: float = 0.15,
    green_override: bool = False,
    road_continuity: float = 0.7,
    seed: int | None = None,
):
    """
    Returns (building_height: float32[size,size], land_use: uint8[size,size]).

    density: 0-1, how likely a buildable cell near a cluster center is to
        actually become a building.
    cluster_tightness: 0-1, how tightly buildings hug their cluster centers.
        Low = broad soft-edged sprawl; high = sharp compact downtown cores.
        Independent of density.
    green_amount: 0-1, how much green space to add.
    green_override: if True, green space is placed over the whole board
        (buildings and roads included) except water. If False, green only
        fills ground left empty after buildings and roads are placed.
    road_continuity: 0-1, how complete the road grid is.
    """
    size = terrain_height.shape[0]
    rng = np.random.default_rng(seed)
    density = float(np.clip(density, 0.05, 1.0))
    cluster_tightness = float(np.clip(cluster_tightness, 0.01, 1.0))
    green_amount = float(np.clip(green_amount, 0.0, 1.0))
    road_continuity = float(np.clip(road_continuity, 0.0, 1.0))

    centers = _cluster_centers(archetype, size, num_clusters, rng)

    # Higher tightness -> smaller spread -> sharper falloff around each
    # center. The +0.03 keeps spread above zero so the math never blows up.
    spread = 0.5 * (1.0 - cluster_tightness) + 0.03

    yy, xx = np.mgrid[0:size, 0:size]

    # TODO 2: build the intensity field.
    # intensity is "how urban is this cell", 0-1, peaking at the cluster
    # centers. With several centers, each cell takes the STRONGEST nearby
    # center's influence -- not the sum, which would make overlapping
    # clusters unrealistically dense where they meet.
    #
    # For each (cx, cy) in centers:
    #   dist    = distance from every cell to that center, divided by size
    #             (same as Week 1: np.sqrt((xx - cx)**2 + (yy - cy)**2) / size)
    #   falloff = np.exp(-dist / spread)
    #   combine with np.maximum(intensity, falloff)
    intensity = np.zeros((size, size), dtype=np.float32)
    for cx, cy in centers:
        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2) / size
        falloff = np.exp(-dist / spread)
        intensity = np.maximum(intensity, falloff)

    # TODO 3: work out where building is physically possible.
    # Two conditions, both required:
    #   - not water
    #   - not too steep
    # np.gradient(terrain_height) returns the slope in y and x. Combine
    # them into a magnitude with np.sqrt(gx**2 + gy**2), then call a cell
    # buildable when that magnitude is below 0.08.
    # Remember `~` is "not" for boolean arrays.
    gy, gx = np.gradient(terrain_height)
    slope_mag = np.sqrt(gx**2 + gy**2)
    buildable = (~is_water) & (slope_mag < 0.08)

    # TODO 4: pick the building cells.
    # Same probability roll as Week 1 -- a random number per cell, compared
    # against intensity * density -- but now ALSO requiring the cell to be
    # buildable.
    is_building = buildable & (rng.random((size, size)) < intensity * density)

    # GIVEN: building heights. The blur makes neighboring buildings similar
    # in height, so you get coherent low-rise districts and tall downtown
    # cores instead of a random jumble.
    height_noise = rng.normal(loc=1.0, scale=0.4, size=(size, size))
    height_noise = gaussian_filter(height_noise, sigma=1.5)
    raw_height = np.clip(intensity * 8.0 * height_noise, 0.3, None)
    building_height = np.where(is_building, raw_height, 0.0).astype(np.float32)

    land_use = np.full((size, size), LAND_EMPTY, dtype=np.uint8)
    land_use[is_building] = LAND_BUILDING

    # TODO 5: lay down roads.
    # Candidate road cells are every `spacing`-th row and every
    # `spacing`-th column. A candidate actually becomes a road only if it
    # is buildable AND still empty (don't pave over buildings) AND kept by
    # the road_keep mask.
    #
    # The loop below walks the two slices for you -- rows, then columns.
    # Inside, build `mask` from those three conditions applied to the
    # slice, then write LAND_ROAD into the masked cells.
    #
    # Note the sub/assign-back dance: numpy slice assignment through a
    # boolean mask needs the intermediate variable to write correctly.
    road_keep = _road_keep_mask(size, rng, road_continuity)
    spacing = max(size // 16, 4)
    for axis_slice in (np.s_[::spacing, :], np.s_[:, ::spacing]):
        mask = (buildable[axis_slice] & (land_use[axis_slice] == LAND_EMPTY)
                & road_keep[axis_slice])
        sub = land_use[axis_slice]
        sub[mask] = LAND_ROAD
        land_use[axis_slice] = sub

    # TODO 6: add green space -- two different behaviors.
    if green_override:
        # Override mode: green goes anywhere that isn't water, replacing
        # buildings and roads. Set those cells to LAND_GREEN, and don't
        # forget to zero out building_height there too -- otherwise you'd
        # leave invisible buildings standing inside a park.
        green_mask = (~is_water) & (rng.random((size, size)) < green_amount)
        land_use[green_mask] = LAND_GREEN
        building_height[green_mask] = 0.0
    else:
        # Normal mode: green only fills cells that are still LAND_EMPTY.
        # Buildings and roads are left alone.
        empty_mask = (land_use == LAND_EMPTY) & (~is_water)
        green_mask = empty_mask & (rng.random((size, size)) < green_amount)
        land_use[green_mask] = LAND_GREEN

    return building_height, land_use
