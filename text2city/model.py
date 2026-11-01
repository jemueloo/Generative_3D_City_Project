"""
Text -> city-parameter model.

    sentence
      -> frozen sentence embedding (embedders.py; NOT trained)
      -> small shared MLP "trunk"            <- trained
      -> one output "head" per parameter     <- trained
           categorical: one logit per choice (softmax / cross-entropy)
           continuous:  one number, squashed to 0-1 by a sigmoid, then
                        rescaled into that parameter's [low, high]
           binary:      one logit (sigmoid > 0.5 means True)

Why this shape: the embedding model already knows English. All we train is
the mapping from "meaning of the sentence" to "this generator's 10
numbers" -- a few tens of thousands of weights, which is why it trains in
about a minute on a laptop CPU.

Why one head per parameter: each head reads the same trunk features but
makes its own prediction, so density and relief are free to vary
independently instead of being forced to move together.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from text2city.schema import PARAM_SCHEMA


class TextToCityParams(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int = 128):
        super().__init__()

        # shared trunk: embedding -> hidden -> hidden
        self.trunk = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # one head per parameter, stored by name so training code can find
        # each head without depending on order
        self.heads = nn.ModuleDict()
        for spec in PARAM_SCHEMA:
            # TODO 5: one nn.Linear(in_features, out_features) per parameter,
            # reading the trunk's hidden_dim features. How many outputs?
            # A categorical head needs one score per choice; a continuous or
            # binary head needs a single number.
            if spec.kind == "categorical":
                self.heads[spec.name] = nn.Linear(hidden_dim, len(spec.choices))
            elif spec.kind in ("continuous", "binary"):
                self.heads[spec.name] = nn.Linear(hidden_dim, 1)
            else:
                raise ValueError(f"unknown parameter kind: {spec.kind}")

    def forward(self, embedding: torch.Tensor) -> dict:
        """
        embedding: (batch, embedding_dim). Returns {param_name: raw output}
        -- logits for categorical/binary, unsquashed values for continuous.
        decode_outputs() turns these into real parameter values.
        """
        # TODO 6: run the embedding through the shared trunk ONCE, then hand
        # those same features to every head. Return {name: head_output}.
        # Hint: self.heads.items() gives you (name, head) pairs.
        features = self.trunk(embedding)
        return {name: head(features) for name, head in self.heads.items()}


def decode_outputs(raw_outputs: dict) -> dict:
    """
    One example's raw outputs (no batch dimension) -> real parameter values:
        categorical -> the name of the highest-scoring choice
        continuous  -> sigmoid(raw), rescaled into [low, high]
        binary      -> sigmoid(raw) > 0.5
    """
    result = {}
    schema_by_name = {p.name: p for p in PARAM_SCHEMA}
    for name, raw in raw_outputs.items():
        spec = schema_by_name[name]
        if spec.kind == "categorical":
            result[name] = spec.choices[torch.argmax(raw).item()]
        elif spec.kind == "continuous":
            low, high = spec.value_range
            result[name] = low + torch.sigmoid(raw).item() * (high - low)
        elif spec.kind == "binary":
            result[name] = torch.sigmoid(raw).item() > 0.5
    return result


def decode_batch(raw_outputs: dict, index: int) -> dict:
    """decode_outputs() for one row of a batched forward() output."""
    return decode_outputs({name: raw[index] for name, raw in raw_outputs.items()})
