#!/usr/bin/env python3
"""
Evaluate a trained checkpoint, two ways -- deliberately kept separate:

  1. Held-out synthetic sentences: same phrase banks as training, but
     examples the model never trained on. Tells you if it learned the
     mapping at all.
  2. Hand-written sentences no template could have produced: tells you if
     it generalises to how a person actually types. This is the number
     that matters. A model that aces (1) and fails (2) memorised the
     templates instead of learning the task.

    python text2city/evaluate.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from text2city.schema import PARAM_SCHEMA, PARAM_BY_NAME  # noqa: E402
from text2city.dataset import load_rows, build_targets, train_val_split  # noqa: E402
from text2city.embedders import load_embedder, embedder_filename  # noqa: E402
from text2city.model import TextToCityParams, decode_batch  # noqa: E402
from text2city.freshness import check_staleness  # noqa: E402


# Hand-written sentences, each with what a person would EXPECT the model to
# get right. Only list the parameters the sentence actually talks about.
#   categorical -> the exact choice, e.g. "polycentric"
#   continuous  -> "high" or "low" (relative to that parameter's range),
#                  or an int for num_clusters (within 0.75 counts)
#   binary      -> True / False
HANDWRITTEN_EXAMPLES = [
    ("a huge sprawling metropolis packed with skyscrapers", {"density": "high"}),
    ("a quiet little town on flat land with hardly any buildings", {"terrain": "flat", "density": "low"}),
    ("a city in the hills with lots of parks everywhere", {"terrain": "hills", "green_amount": "high"}),
    ("a city with three separate business hubs", {"city_archetype": "polycentric", "num_clusters": 3}),
    ("a long thin city stretched along one road", {"city_archetype": "linear"}),
    ("a sparse town with almost no roads", {"density": "low", "road_continuity": "low"}),
    # TODO 10: add at least THREE sentences of your own below, phrased the way
    # a person would actually type -- not copied from the phrase banks in
    # generate_dataset.py. Try to catch the model out.
    ("a tiny village in the middle of nowhere, mostly fields and trees, hardly any streets",
     {"density": "low", "green_amount": "high", "road_continuity": "low"}),
    ("flat as a pancake and totally built over, skyscrapers everywhere", {"terrain": "flat", "density": "high"}),
    ("a pair of busy downtown areas on rolling hills", {"terrain": "hills", "city_archetype": "polycentric"}),
    ("a crowded, gridlocked metropolis with a well connected street grid and almost no parks",
     {"density": "high", "road_continuity": "high", "green_amount": "low"}),
]


def expectation_met(spec, predicted, expected) -> bool:
    if spec.kind in ("categorical", "binary"):
        return predicted == expected
    low, high = spec.value_range
    if expected == "high":
        return predicted > low + 0.6 * (high - low)
    if expected == "low":
        return predicted < low + 0.4 * (high - low)
    return abs(predicted - float(expected)) <= 0.75


def load_checkpoint(checkpoint_dir: str, dataset_path: str):
    ckpt_dir = Path(checkpoint_dir)
    ckpt = torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=False)
    check_staleness(ckpt.get("fingerprint"), dataset_path)
    model = TextToCityParams(embedding_dim=ckpt["embedding_dim"], hidden_dim=ckpt["hidden_dim"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    kind = ckpt["embedder_kind"]
    embedder = load_embedder(kind, str(ckpt_dir / embedder_filename(kind)))
    return model, embedder


def predict(model, embedder, texts: list[str]) -> list[dict]:
    with torch.no_grad():
        raw = model(torch.as_tensor(embedder.encode(texts)))
    return [decode_batch(raw, i) for i in range(len(texts))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default="text2city/checkpoint")
    parser.add_argument("--dataset", type=str, default="data/text2city_dataset.csv")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model, embedder = load_checkpoint(args.checkpoint, args.dataset)

    # ---- 1. held-out synthetic examples ----
    rows = load_rows(args.dataset)
    targets = build_targets(rows)
    _, val_idx = train_val_split(len(rows), args.val_fraction, seed=args.seed)
    preds = predict(model, embedder, [rows[i]["text"] for i in val_idx])

    print("=== 1. held-out synthetic examples ===")
    for spec in PARAM_SCHEMA:
        if spec.kind in ("categorical", "binary"):
            # TODO 9a: accuracy = the fraction of validation rows where the
            # prediction matches the label. zip(val_idx, preds) pairs each row
            # index i with its prediction p. The CSV stores everything as text,
            # so compare str(p[spec.name]) with rows[i][spec.name].
            # np.mean of a list of True/False gives the fraction that are True.
            accuracy = np.mean([str(p[spec.name]) == rows[i][spec.name] for i, p in zip(val_idx, preds)])
            print(f"  {spec.name:20s} accuracy = {accuracy:.3f}")
        else:
            # mean absolute error in the parameter's real units
            predicted = np.array([p[spec.name] for p in preds])
            actual = targets[f"{spec.name}__raw"][val_idx]
            # TODO 9b: mean absolute error between predicted and actual.
            mae = np.mean(np.abs(predicted - actual))
            print(f"  {spec.name:20s} MAE      = {mae:.4f}")

    # ---- 2. hand-written sentences ----
    print("\n=== 2. hand-written sentences (the generalisation test) ===")
    hw_preds = predict(model, embedder, [text for text, _ in HANDWRITTEN_EXAMPLES])
    hits = total = 0
    for (text, expected), pred in zip(HANDWRITTEN_EXAMPLES, hw_preds):
        print(f'\n  "{text}"')
        for name, want in expected.items():
            if name not in PARAM_BY_NAME:
                print(f"      {name:20s} (not a parameter -- check the spelling)")
                continue
            if name == "terrain" and want not in PARAM_BY_NAME["terrain"].choices:
                print(f"      {name:20s} skipped: your citygen has no {want!r} terrain")
                continue
            got = pred[name]
            ok = expectation_met(PARAM_BY_NAME[name], got, want)
            hits += ok
            total += 1
            shown = round(got, 2) if isinstance(got, float) else got
            print(f"      {'PASS' if ok else 'MISS'}  {name:20s} wanted {want!r:14} got {shown!r}")
    if total:
        print(f"\nhand-written expectations met: {hits}/{total} ({hits / total:.0%})")


if __name__ == "__main__":
    main()
