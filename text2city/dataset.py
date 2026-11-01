"""
Loads the synthetic (sentence -> parameters) CSV and turns it into the
arrays the training loop needs:

    categorical -> int64 class index   (which choice, as a number)
    continuous  -> float32 in [0, 1]   (rescaled, to match the model's sigmoid output)
    binary      -> float32 0.0 or 1.0
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from text2city.schema import PARAM_SCHEMA  # noqa: E402


def load_rows(csv_path: str) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def build_targets(rows: list[dict]) -> dict:
    """
    Returns {param_name: np.ndarray of shape (N,)}, one array per parameter.

    For continuous parameters it ALSO returns the original, un-rescaled
    values under "<name>__raw" -- handy for reporting errors in real units
    (density off by 0.04) rather than rescaled ones.

    Everything in the CSV is a string, so convert as you go.
    """
    targets = {}
    for spec in PARAM_SCHEMA:
        col = [r[spec.name] for r in rows]
        if spec.kind == "categorical":
            # TODO 4a: each value is a choice name like "hills". Store its
            # POSITION in spec.choices instead (spec.choices.index(v)), as int64.
            targets[spec.name] = np.array([spec.choices.index(v) for v in col], dtype=np.int64)
        elif spec.kind == "continuous":
            raw = np.array([float(v) for v in col], dtype=np.float32)
            low, high = spec.value_range
            # TODO 4b: rescale raw from [low, high] to [0, 1], as float32.
            # The model's continuous heads end in a sigmoid, which can only
            # output 0-1 -- so the targets have to live there too.
            targets[spec.name] = ((raw - low) / (high - low)).astype(np.float32)
            targets[f"{spec.name}__raw"] = raw
        elif spec.kind == "binary":
            # TODO 4c: the CSV holds the TEXT "True" or "False". Store 1.0 / 0.0, as float32.
            targets[spec.name] = np.array([v == "True" for v in col], dtype=np.float32)
    return targets


def train_val_split(n: int, val_fraction: float = 0.15, seed: int = 0):
    """Shuffle 0..n-1 and cut off the last val_fraction for validation. Returns (train_idx, val_idx)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = int(n * val_fraction)
    return idx[n_val:], idx[:n_val]
