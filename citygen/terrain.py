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
TERRAIN_ARCHETYPES = (
    "flat",
    "hills",
    # ADD MORE ARCHETYPES HERE IN LATER WEEKS:
    # "mountains", "valley", "coastal_harbor", "peninsula", "river",
)


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


# --------------------------------------------------------------------
# YOUR CODE -- generate_terrain
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

    # has_water marks archetypes that are *supposed* to contain water by
    # default (coastlines, rivers). flat and hills are land-locked, so this
    # stays False for both this week.
    has_water = False

    # Each branch sets TWO things:
    #   height       -- the SHAPE of the terrain (before normalizing)
    #   height_scale -- how much vertical variation it has, applied AFTER
    #                   normalizing (see the note above -- scaling before
    #                   normalize does nothing at all)

    if archetype == "flat":
        # TODO 1: nearly level ground.
        # The shape is just the plain noise -- no carving needed.
        # Then pick a SMALL height_scale so the result stays nearly level.
        # Something around 0.2-0.3 gives recognizably flat ground.
        height = base
        height_scale = 0.25

    elif archetype == "hills":
        # TODO 2: gentle rolling hills.
        # Same plain-noise shape, but keep the full vertical range this
        # time -- a height_scale of 1.0 means "don't compress it at all".
        height = base
        height_scale = 1.0

    # ADD MORE ARCHETYPE BRANCHES HERE IN LATER WEEKS
    # (and remember to add the name to TERRAIN_ARCHETYPES at the top).
    # Archetypes like valley or coastal_harbor change the SHAPE -- they
    # subtract from `height` before normalizing -- and can leave
    # height_scale at 1.0.

    else:
        raise ValueError(f"archetype {archetype!r} is listed but not implemented")

    height = _normalize(height)

    # TODO 3: decide which cells are water.
    # Water is decided here, on the normalized heightmap, BEFORE any
    # vertical scaling below. That's deliberate: it keeps "how much water"
    # independent of "how much the terrain varies", so changing relief
    # never accidentally floods or drains the map.
    #
    #   - default_water_level: if this archetype has water by default
    #     (has_water is True), use 0.2; otherwise use -0.01, which is below
    #     every possible height and therefore means "no water at all".
    #   - level: use the caller's water_level if they passed one, else the
    #     default you just computed.
    #   - is_water: True wherever height is at or below that level.
    default_water_level = 0.2 if has_water else -0.01
    level = water_level if water_level is not None else default_water_level
    is_water = height <= level

    # TODO 4: apply the vertical scaling -- both the archetype's own
    # height_scale and the caller's relief knob.
    #
    # Both do the same kind of thing: contrast-adjust the heightmap around
    # its midpoint (0.5). Above 1.0 pushes highs higher and lows lower;
    # below 1.0 squeezes everything toward 0.5 (flatter). Multiply the two
    # together so they compose -- flat terrain with relief=1.8 should still
    # be flatter than hills at relief=1.0.
    #
    # Hint: 0.5 + (height - 0.5) * height_scale * relief, then
    # np.clip(..., 0.0, 1.0) to keep it in range.
    relief = max(relief, 0.0)
    height = np.clip(0.5 + (height - 0.5) * height_scale * relief, 0.0, 1.0)

    return height.astype(np.float32), is_water
