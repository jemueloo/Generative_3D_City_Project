"""
Single source of truth for the 10 city-generator parameters the
text-to-parameter model predicts.

Every other piece (dataset generation, model, training, evaluation, and the
inference script in a later week) imports PARAM_SCHEMA from here instead of
hardcoding parameter names or ranges. If the generator's parameters ever
change, this is the one file to update.

Each parameter is one of three kinds:
    categorical -- a fixed set of named choices (e.g. terrain archetype)
    continuous  -- a float in [low, high]
    binary      -- True/False

`default` is what the model should predict when a parameter is NOT
mentioned in the text at all (see generate_dataset.py). That keeps the
model predictable on short, partial prompts instead of guessing wildly
for anything the user didn't specify.

Terrain choices come from YOUR citygen package, not a fixed list: if your
terrain.py only implements flat and hills, the model only ever learns to
predict flat and hills -- so it can never hand the generator a terrain
type your code doesn't have.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from citygen import TERRAIN_ARCHETYPES, CITY_ARCHETYPES  # noqa: E402
except ImportError as exc:
    raise ImportError(
        "text2city needs your citygen package (terrain.py and buildings.py) to "
        "import cleanly, because the model learns exactly the terrain and city "
        f"types you implemented. Original error: {exc}"
    ) from exc


@dataclass
class ParamSpec:
    name: str
    kind: str  # "categorical" | "continuous" | "binary"
    default: object
    choices: Optional[list] = None       # required for "categorical"
    value_range: Optional[tuple] = None  # required for "continuous", (low, high)
    # how often a synthetic training sentence mentions this parameter at all
    mention_prob: float = 0.35


_TERRAINS = list(TERRAIN_ARCHETYPES)
_TERRAIN_DEFAULT = "hills" if "hills" in _TERRAINS else _TERRAINS[0]
_CITIES = list(CITY_ARCHETYPES)
_CITY_DEFAULT = "monocentric" if "monocentric" in _CITIES else _CITIES[0]


PARAM_SCHEMA: list[ParamSpec] = [
    ParamSpec(
        name="terrain", kind="categorical", default=_TERRAIN_DEFAULT,
        choices=_TERRAINS,
        mention_prob=0.8,
    ),
    ParamSpec(
        name="city_archetype", kind="categorical", default=_CITY_DEFAULT,
        choices=_CITIES,
        mention_prob=0.65,
    ),
    ParamSpec(
        name="num_clusters", kind="continuous", default=1.0,
        value_range=(1.0, 6.0),
        mention_prob=0.25,
    ),
    ParamSpec(
        name="density", kind="continuous", default=0.5,
        value_range=(0.0, 1.0),
        mention_prob=0.55,
    ),
    ParamSpec(
        name="cluster_tightness", kind="continuous", default=0.5,
        value_range=(0.0, 1.0),
        mention_prob=0.3,
    ),
    ParamSpec(
        name="relief", kind="continuous", default=1.0,
        value_range=(0.0, 2.0),
        mention_prob=0.3,
    ),
    ParamSpec(
        name="water_level", kind="continuous", default=0.2,
        value_range=(0.0, 0.6),
        mention_prob=0.3,
    ),
    ParamSpec(
        name="green_amount", kind="continuous", default=0.15,
        value_range=(0.0, 1.0),
        mention_prob=0.45,
    ),
    ParamSpec(
        name="green_override", kind="binary", default=False,
        mention_prob=0.1,
    ),
    ParamSpec(
        name="road_continuity", kind="continuous", default=0.7,
        value_range=(0.0, 1.0),
        mention_prob=0.25,
    ),
]

PARAM_NAMES = [p.name for p in PARAM_SCHEMA]
PARAM_BY_NAME = {p.name: p for p in PARAM_SCHEMA}

CATEGORICAL_PARAMS = [p for p in PARAM_SCHEMA if p.kind == "categorical"]
CONTINUOUS_PARAMS = [p for p in PARAM_SCHEMA if p.kind == "continuous"]
BINARY_PARAMS = [p for p in PARAM_SCHEMA if p.kind == "binary"]


def default_vector() -> dict:
    """The full parameter dict when nothing at all is mentioned."""
    return {p.name: p.default for p in PARAM_SCHEMA}


def to_generator_kwargs(params: dict) -> dict:
    """
    Map this schema's parameter names onto generate_city.py's build_city()
    keyword arguments.

    water_level needs special handling: the model always predicts SOME
    number for it, but generate_terrain()'s own default depends on the
    archetype (land-locked ones default to no water at all). So if the
    prediction sits near the schema default -- meaning the text probably
    said nothing about water -- pass None and let generate_terrain() pick
    the right per-archetype default. Otherwise an unmentioned water level
    can flood a third of a mountain map.
    """
    water_level = params["water_level"]
    water_spec = PARAM_BY_NAME["water_level"]
    water_level_kwarg = None if abs(water_level - water_spec.default) < 0.03 else water_level

    return {
        "terrain_archetype": params["terrain"],
        "city_archetype": params["city_archetype"],
        "clusters": int(round(params["num_clusters"])),
        "density": params["density"],
        "cluster_tightness": params["cluster_tightness"],
        "relief": params["relief"],
        "water_level": water_level_kwarg,
        "green_amount": params["green_amount"],
        "green_override": bool(params["green_override"]),
        "road_continuity": params["road_continuity"],
    }
