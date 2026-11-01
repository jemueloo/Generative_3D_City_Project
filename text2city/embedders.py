"""
Sentence-embedding backends -- turn a list of sentences into a fixed-size
float array. Same interface for both: fit() on the training texts (a no-op
for the pretrained one), then encode() any list of strings.

  "sentence-transformer" (what we use): all-MiniLM-L6-v2, a pretrained
      model that already understands English. It places "surrounded by
      peaks" close to "surrounded by mountains" even though only one of
      them is in our dataset. Needs `sentence-transformers` and a one-time
      ~80MB download (cached, offline afterwards). It is FROZEN: we never
      train it, only the small model on top.

  "tfidf" (offline fallback): classic word counts. Only knows words it saw
      while fitting, so it can't handle a synonym it never met -- but it
      needs no download, which makes it a useful backup.
"""
from __future__ import annotations

import pickle

import numpy as np


class TfidfEmbedder:
    kind = "tfidf"

    def __init__(self, max_features: int = 1200):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        self._fitted = False

    def fit(self, texts: list[str]):
        self.vectorizer.fit(texts)
        self._fitted = True
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder.fit() must be called before encode()")
        return self.vectorizer.transform(texts).toarray().astype(np.float32)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    @classmethod
    def load(cls, path: str) -> "TfidfEmbedder":
        obj = cls.__new__(cls)
        with open(path, "rb") as f:
            obj.vectorizer = pickle.load(f)
        obj._fitted = True
        return obj


class SentenceTransformerEmbedder:
    kind = "sentence-transformer"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def fit(self, texts: list[str]):
        return self  # pretrained and frozen -- nothing to fit

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, convert_to_numpy=True), dtype=np.float32)

    def save(self, path: str):
        # nothing to save but the name -- the weights live in the download cache
        with open(path, "w") as f:
            f.write(self.model_name)

    @classmethod
    def load(cls, path: str) -> "SentenceTransformerEmbedder":
        with open(path) as f:
            return cls(f.read().strip())


def get_embedder(kind: str, **kwargs):
    if kind == "tfidf":
        return TfidfEmbedder(**kwargs)
    if kind == "sentence-transformer":
        return SentenceTransformerEmbedder(**kwargs)
    raise ValueError(f"unknown embedder kind {kind!r}, choose 'tfidf' or 'sentence-transformer'")


def embedder_filename(kind: str) -> str:
    return "embedder.pkl" if kind == "tfidf" else "embedder.txt"


def load_embedder(kind: str, path: str):
    if kind == "tfidf":
        return TfidfEmbedder.load(path)
    if kind == "sentence-transformer":
        return SentenceTransformerEmbedder.load(path)
    raise ValueError(f"unknown embedder kind {kind!r}")
