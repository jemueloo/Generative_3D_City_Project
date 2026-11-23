"""
Procedural terrain generation.

Produces a heightmap using multi-octave blurred noise, then shapes it into
a named archetype. Returns (terrain_height, is_water) -- the shared contract
every downstream part of the pipeline depends on.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

# Every archetype this module knows how to build.
# generate_city.py reads this tuple directly to build its --terrain menu,
# so adding a new archetype here (plus a branch below) makes it instantly
# available on the command line. No other file needs to change.
#
# Names match the phrase banks in text2city/generate_dataset.py.
TERRAIN_ARCHETYPES = (
    "flat",
    "hills",
    "mountains",
    "valley",
    "coastal_harbor",
    "peninsula",
    "river",
)

# Archetypes that contain water unless the caller passes a lower level.
# flat, hills, mountains and valley stay dry by default.
_HAS_WATER = {"coastal_harbor", "peninsula", "river"}

# Fraction of the normalized heightmap that is water, before relief is applied.
# Tuned so a harbor is a coast, a peninsula is surrounded on three sides,
# and a river is a channel rather than a lake.
_WATER_DEFAULTS = {
    "coastal_harbor": 0.42,
    "peninsula": 0.30,
    "river": 0.28,
}


# --------------------------------------------------------------------
# GIVEN -- helpers, no changes needed
# --------------------------------------------------------------------

def _normalize(arr: np.ndarray) -> np.ndarray:
    """Rescale an array so its values span 0.0 to 1.0."""
    arr = arr - arr.min()
    peak = arr.max()
    return arr / peak if peak > 1e-8 else arr


def _fractal_noise(size: int, rng: np.random.Generator, base_scale: float,
                   octaves: int = 4, persistence: float = 0.5) -> np.ndarray:
    """
    Sum of several blurred-noise layers at decreasing scale and amplitude.

    Each "octave" is random noise blurred by a smaller amount than the last
    and added in at a smaller amplitude. The result has big rolling shapes
    AND fine surface detail, which a single blurred noise field can't do.
    Output is normalized to 0.0 - 1.0.
    """
    height = np.zeros((size, size), dtype=np.float32)
    amplitude = 1.0
    total_amp = 0.0
    for octave in range(octaves):
        sigma = max(base_scale / (2 ** octave), 1.0)
        layer = rng.normal(size=(size, size)).astype(np.float32)
        layer = gaussian_filter(layer, sigma=sigma)
        height += amplitude * layer
        total_amp += amplitude
        amplitude *= persistence
    height /= total_amp
    return _normalize(height)


def _inland_along(size: int, yy: np.ndarray, xx: np.ndarray, edge: int):
    """
    Coordinates relative to one map edge.

    inland is 0 on the chosen edge and 1 on the opposite edge.
    along runs along that edge, in cell units.
    """
    span = float(size - 1)
    if edge == 0:      # water / attachment along y = 0
        return yy / span, xx
    if edge == 1:      # along y = size - 1
        return (span - yy) / span, xx
    if edge == 2:      # along x = 0
        return xx / span, yy
    return (span - xx) / span, yy


def _river_centerline(size: int, rng: np.random.Generator) -> np.ndarray:
    """A meandering path from one edge of the map to another.

    The bends are a sideways random walk, strongest in the middle, then
    smoothed. That reads as a river, not a straight canal or a single sine.
    """
    start = int(rng.integers(0, 4))
    # Opposite edge most of the time, so the river actually crosses the map.
    end = (start + 2) % 4 if rng.random() < 0.7 else (start + int(rng.choice([1, 3]))) % 4

    def on_edge(edge: int, t: float):
        t = float(np.clip(t, 0.12, 0.88))
        if edge == 0:
            return np.array([t * (size - 1), 0.0], np.float32)
        if edge == 1:
            return np.array([t * (size - 1), size - 1.0], np.float32)
        if edge == 2:
            return np.array([0.0, t * (size - 1)], np.float32)
        return np.array([size - 1.0, t * (size - 1)], np.float32)

    p0 = on_edge(start, float(rng.uniform(0.3, 0.7)))
    p1 = on_edge(end, float(rng.uniform(0.3, 0.7)))
    n = 9
    pts = np.array([(1 - i / (n - 1)) * p0 + (i / (n - 1)) * p1 for i in range(n)], np.float32)
    direction = p1 - p0
    direction = direction / (np.linalg.norm(direction) + 1e-6)
    perp = np.array([-direction[1], direction[0]], np.float32)
    wander = 0.0
    for i in range(1, n - 1):
        wander = 0.72 * wander + float(rng.normal(0, size * 0.07))
        envelope = np.sin(np.pi * i / (n - 1))
        pts[i] += perp * wander * envelope
    for _ in range(3):
        smoothed = pts.copy()
        smoothed[1:-1] = 0.22 * pts[:-2] + 0.56 * pts[1:-1] + 0.22 * pts[2:]
        pts = smoothed
    pts[:, 0] = np.clip(pts[:, 0], 1, size - 2)
    pts[:, 1] = np.clip(pts[:, 1], 1, size - 2)
    # Force the ends back onto the border so the water touches the edge.
    pts[0] = p0
    pts[-1] = p1

    dense = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        steps = max(int(np.linalg.norm(b - a)), 1)
        for s in range(1, steps + 1):
            dense.append(a + (b - a) * (s / steps))
    line = np.array(dense, np.float32)
    if len(line) > 350:
        line = line[:: int(np.ceil(len(line) / 350))]
    return line


# --------------------------------------------------------------------
# generate_terrain
# --------------------------------------------------------------------

def generate_terrain(
    size: int,
    archetype: str = "hills",
    seed: int | None = None,
    relief: float = 1.0,
    water_level: float | None = None,
):
    """
    Returns (terrain_height: float32[size,size], is_water: bool[size,size]).

    archetype: which terrain shape to build (see TERRAIN_ARCHETYPES).
    relief: how much the terrain varies. 1.0 = as designed. Lower (0.3)
        flattens toward gentle ground; higher (1.8) exaggerates peaks and
        valleys. Must NOT change how much water there is -- that's
        water_level's job, and the two need to stay independent.
    water_level: 0-1 threshold below which ground counts as water. Higher =
        more water. If None, use the archetype's own default.
    """
    if archetype not in TERRAIN_ARCHETYPES:
        raise ValueError(f"unknown terrain archetype {archetype!r}, choose from {TERRAIN_ARCHETYPES}")

    rng = np.random.default_rng(seed)
    base = _fractal_noise(size, rng, base_scale=size / 6)
    yy, xx = np.mgrid[0:size, 0:size]
    yy = yy.astype(np.float32)
    xx = xx.astype(np.float32)

    # has_water marks archetypes that contain water by default
    # (coast, peninsula, river). The others are land-locked.
    has_water = archetype in _HAS_WATER

    # Each branch sets TWO things:
    #   height       -- the SHAPE of the terrain (before normalizing)
    #   height_scale -- how much vertical variation it has, applied AFTER
    #                   normalizing. Scaling before normalize does nothing.

    if archetype == "flat":
        height = base
        height_scale = 0.25

    elif archetype == "hills":
        height = base
        height_scale = 1.0

    elif archetype == "mountains":
        # Ridged noise makes crests. Raising it to a power keeps the
        # valleys broad and the peaks sharp, which is the difference
        # from hills (hills are the same noise, just less tall).
        ridged = 1.0 - np.abs(2.0 * base - 1.0)
        peaks = np.zeros((size, size), dtype=np.float32)
        for _ in range(int(rng.integers(3, 6))):
            px = float(rng.uniform(0.18, 0.82) * size)
            py = float(rng.uniform(0.18, 0.82) * size)
            sigma = float(size * rng.uniform(0.05, 0.11))
            peaks += np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2.0 * sigma ** 2))
        height = 0.50 * (ridged ** 2.2) + 0.50 * _normalize(peaks)
        height_scale = 1.65

    elif archetype == "valley":
        # A soft trench along a gently tilted axis. The floor stays
        # buildable; the walls rise on either side. Dry by default.
        angle = float(rng.uniform(-0.7, 0.7))
        xr = (xx - size / 2) * np.cos(angle) + (yy - size / 2) * np.sin(angle)
        trench = np.exp(-((xr / (size * 0.15)) ** 2))
        height = (1.0 - trench) * (0.55 + 0.45 * base) + trench * (0.04 + 0.12 * base)
        height_scale = 1.25

    elif archetype == "coastal_harbor":
        # A shelf of sea along one edge, with a deeper bay cut into it.
        # The shelf, not the bay alone, is what makes the water a coastline.
        edge = int(rng.integers(0, 4))
        inland, along = _inland_along(size, yy, xx, edge)
        bay_at = float(rng.uniform(0.32, 0.68) * size)
        bay_dist = np.abs(along - bay_at) / size
        bay = np.exp(-((bay_dist / 0.16) ** 2)) * np.exp(-((inland / 0.40) ** 2))
        shelf = np.clip((inland - 0.28) / 0.50, 0.0, 1.0)
        height = shelf * 0.90 + base * 0.16 * (0.25 + shelf) - bay * 0.50
        height_scale = 1.15

    elif archetype == "peninsula":
        # Half an oval tied to one edge and reaching well inland.
        # The three open sides are the low part of the map, so they flood.
        edge = int(rng.integers(0, 4))
        inland, along = _inland_along(size, yy, xx, edge)
        cx = float(rng.uniform(0.40, 0.60))
        nx = (along / size - cx) / 0.42
        ny = inland / 0.88
        radius = np.sqrt(nx ** 2 + ny ** 2)
        # Solid land inside the oval, with a short shore so the water
        # line follows the peninsula instead of swallowing it.
        shore = np.clip((1.02 - radius) / 0.16, 0.0, 1.0)
        height = shore * (0.62 + 0.38 * base) + (1.0 - shore) * base * 0.03
        height_scale = 0.80

    elif archetype == "river":
        # One channel from edge to edge. The bed is the only low ground,
        # so the default water stays in the river instead of flooding.
        line = _river_centerline(size, rng)
        dx = xx[..., None] - line[None, None, :, 0]
        dy = yy[..., None] - line[None, None, :, 1]
        dist = np.sqrt(dx * dx + dy * dy).min(axis=-1)
        width = float(size * rng.uniform(0.032, 0.048))
        channel = np.exp(-((dist / width) ** 2))
        banks = np.exp(-((dist / (width * 3.2)) ** 2))
        height = 0.22 * base + 0.62 * (1.0 - banks) - channel * 1.05
        height_scale = 0.9

    else:
        raise ValueError(f"archetype {archetype!r} is listed but not implemented")

    height = _normalize(height)

    # Water is decided on the normalized heightmap, BEFORE vertical
    # scaling. Changing relief must not flood or drain the map.
    if not has_water:
        default_water_level = -0.01
    else:
        default_water_level = _WATER_DEFAULTS.get(archetype, 0.2)
    level = water_level if water_level is not None else default_water_level
    is_water = height <= level

    relief = max(relief, 0.0)
    height = np.clip(0.5 + (height - 0.5) * height_scale * relief, 0.0, 1.0)

    return height.astype(np.float32), is_water
