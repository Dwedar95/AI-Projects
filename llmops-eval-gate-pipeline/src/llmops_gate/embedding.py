"""Deterministic text embeddings by signed feature hashing.

This is a stand-in for a learned embedding model so the pipeline runs offline.
Tokens (and optionally bigrams) are hashed with BLAKE2 (stable across processes, unlike the
built-in ``hash``) into a fixed number of buckets with a sign bit, then the
vector is L2-normalised. It captures vocabulary, not semantics.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens."""
    return _TOKEN.findall(text.lower())


def estimate_tokens(text: str) -> int:
    """Rough LLM token estimate (about four characters per token), minimum one."""
    return max(1, len(text) // 4)


@lru_cache(maxsize=65536)
def _bucket(feature: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    return value % dim, 1.0 if (value >> 63) & 1 else -1.0


class HashingEmbedder:
    """Maps text to a unit-length float32 vector of size ``dim``."""

    def __init__(self, dim: int = 256, use_bigrams: bool = False) -> None:
        if dim < 8:
            raise ValueError("dim must be at least 8")
        self.dim = dim
        self.use_bigrams = use_bigrams

    def embed_one(self, text: str) -> np.ndarray:
        tokens = tokenize(text)
        features = list(tokens)
        if self.use_bigrams:
            features += [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
        vector = np.zeros(self.dim, dtype=np.float32)
        for feature in features:
            index, sign = _bucket(feature, self.dim)
            vector[index] += sign
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else vector

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch. Returns an array of shape ``(len(texts), dim)``."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self.embed_one(t) for t in texts])
