#!/usr/bin/env python3
"""
Train the text-to-city-parameter model.

    python text2city/train.py                       # sentence-transformer (default)
    python text2city/train.py --embedder tfidf      # offline fallback

Saves text2city/checkpoint/model.pt (trunk + heads) and the embedder file,
so evaluate.py can load everything from one place.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from text2city.schema import PARAM_SCHEMA  # noqa: E402
from text2city.dataset import load_rows, build_targets, train_val_split  # noqa: E402
from text2city.embedders import get_embedder, embedder_filename  # noqa: E402
from text2city.model import TextToCityParams  # noqa: E402
from text2city.freshness import fingerprint_sources  # noqa: E402


def compute_class_weights(targets: dict, train_idx) -> dict:
    """
    Inverse-frequency weights for each categorical parameter.

    A categorical default ("monocentric", "hills") is ALSO one of its real
    choices, so every sentence that doesn't mention it adds to that class.
    Left alone, the model learns "when unsure, say monocentric". Weighting
    rare classes up removes that shortcut.
    """
    weights = {}
    for spec in PARAM_SCHEMA:
        if spec.kind != "categorical":
            continue
        t = targets[spec.name][train_idx]
        counts = np.array([np.sum(t == i) for i in range(len(spec.choices))], dtype=np.float32)
        inv = 1.0 / np.maximum(counts, 1)
        weights[spec.name] = torch.tensor(inv / inv.sum() * len(spec.choices), dtype=torch.float32)
    return weights


def compute_loss(raw_outputs: dict, targets: dict, class_weights: dict) -> torch.Tensor:
    """
    Sum of one loss per parameter, each picked to match the head's kind:

        categorical -> cross-entropy on the logits (with class weights)
        continuous  -> mean squared error between sigmoid(output) and the 0-1 target
        binary      -> binary cross-entropy on the logit

    raw_outputs[name] has shape (batch, k); continuous and binary heads have
    k=1, so .squeeze(-1) them to (batch,) to line up with their targets.
    """
    total = torch.tensor(0.0)
    for spec in PARAM_SCHEMA:
        pred = raw_outputs[spec.name]
        target = torch.as_tensor(targets[spec.name])
        # TODO 7: pick the loss that matches this head's kind (docstring above).
        # All three live in nn.functional:
        #   cross_entropy(logits, class_indices, weight=class_weights[spec.name])
        #   mse_loss(prediction, target)      <- squash pred with torch.sigmoid first
        #   binary_cross_entropy_with_logits(logit, target)
        if spec.kind == "categorical":
            loss = nn.functional.cross_entropy(pred, target, weight=class_weights[spec.name])
        elif spec.kind == "continuous":
            loss = nn.functional.mse_loss(torch.sigmoid(pred.squeeze(-1)), target)
        else:  # binary
            loss = nn.functional.binary_cross_entropy_with_logits(pred.squeeze(-1), target)
        total = total + loss
    return total


@torch.no_grad()
def evaluate_split(model, embeddings, targets, idx) -> dict:
    """Accuracy for categorical/binary heads, MAE in real units for continuous ones."""
    model.eval()
    raw = model(torch.as_tensor(embeddings[idx]))
    metrics = {}
    for spec in PARAM_SCHEMA:
        target = targets[spec.name][idx]
        if spec.kind == "categorical":
            pred = torch.argmax(raw[spec.name], dim=1).numpy()
            metrics[spec.name] = float((pred == target).mean())
        elif spec.kind == "continuous":
            low, high = spec.value_range
            pred = low + torch.sigmoid(raw[spec.name].squeeze(-1)).numpy() * (high - low)
            metrics[spec.name] = float(np.mean(np.abs(pred - targets[f"{spec.name}__raw"][idx])))
        else:
            pred = (torch.sigmoid(raw[spec.name].squeeze(-1)) > 0.5).numpy().astype(np.float32)
            metrics[spec.name] = float((pred == target).mean())
    model.train()
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="data/text2city_dataset.csv")
    parser.add_argument("--embedder", choices=["sentence-transformer", "tfidf"], default="sentence-transformer")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="text2city/checkpoint")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    rows = load_rows(args.dataset)
    texts = [r["text"] for r in rows]
    targets = build_targets(rows)
    train_idx, val_idx = train_val_split(len(rows), args.val_fraction, seed=args.seed)
    print(f"{len(rows)} examples ({len(train_idx)} train / {len(val_idx)} val)")

    class_weights = compute_class_weights(targets, train_idx)

    print(f"embedding sentences with '{args.embedder}' ...")
    t0 = time.time()
    embedder = get_embedder(args.embedder)
    embedder.fit([texts[i] for i in train_idx])
    embeddings = embedder.encode(texts)
    print(f"  {embeddings.shape[0]} sentences -> dim {embeddings.shape[1]} in {time.time() - t0:.1f}s")

    model = TextToCityParams(embedding_dim=embeddings.shape[1], hidden_dim=args.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"\ntraining for {args.epochs} epochs ...")
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        perm = rng.permutation(train_idx)
        epoch_loss = 0.0
        for start in range(0, len(perm), args.batch_size):
            batch_idx = perm[start:start + args.batch_size]
            batch_emb = torch.as_tensor(embeddings[batch_idx])
            batch_targets = {k: v[batch_idx] for k, v in targets.items()}

            # TODO 8: one optimisation step, in this order:
            #   1. forward pass: the model's raw outputs for batch_emb
            #   2. the loss, from compute_loss(raw, batch_targets, class_weights)
            #   3. clear the previous step's gradients   optimizer.zero_grad()
            #   4. backpropagate                          loss.backward()
            #   5. update the weights                     optimizer.step()
            raw = model(batch_emb)
            loss = compute_loss(raw, batch_targets, class_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(batch_idx)

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            m = evaluate_split(model, embeddings, targets, val_idx)
            print(f"  epoch {epoch:3d}  train_loss={epoch_loss / len(train_idx):.4f}  "
                  f"val terrain_acc={m['terrain']:.3f}  city_acc={m['city_archetype']:.3f}  "
                  f"density_MAE={m['density']:.3f}")

    print(f"\ntraining done in {time.time() - t0:.1f}s")
    print("\nvalidation (accuracy for categorical/binary, MAE in real units for continuous):")
    final = evaluate_split(model, embeddings, targets, val_idx)
    for spec in PARAM_SCHEMA:
        label = "acc" if spec.kind in ("categorical", "binary") else "MAE"
        print(f"  {spec.name:20s} {label}={final[spec.name]:.4f}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "embedding_dim": embeddings.shape[1],
        "hidden_dim": args.hidden_dim,
        "embedder_kind": args.embedder,
        "fingerprint": fingerprint_sources(args.dataset),
    }, out_dir / "model.pt")
    embedder.save(str(out_dir / embedder_filename(args.embedder)))
    print(f"\nsaved checkpoint to {out_dir}/")


if __name__ == "__main__":
    main()
