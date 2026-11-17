"""Okapi BM25 implemented on an in-memory inverted index."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np


class BM25Index:
    """BM25 ranking over tokenised documents.

    score(q, d) = sum over query terms t of
        idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))
    with idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5)), which is never negative.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("k1 must be >= 0 and b must be in [0, 1]")
        self.k1 = k1
        self.b = b
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.idf: dict[str, float] = {}
        self.doc_lengths = np.zeros(0)
        self.avgdl = 0.0

    def __len__(self) -> int:
        return len(self.doc_lengths)

    def fit(self, tokenized_docs: list[list[str]]) -> BM25Index:
        """Build the inverted index."""
        self.postings = {}
        for doc_index, tokens in enumerate(tokenized_docs):
            for term, tf in Counter(tokens).items():
                self.postings.setdefault(term, []).append((doc_index, tf))
        n = len(tokenized_docs)
        self.doc_lengths = np.array([len(tokens) for tokens in tokenized_docs], dtype=float)
        self.avgdl = float(self.doc_lengths.mean()) if n else 0.0
        self.idf = {
            term: math.log(1.0 + (n - len(plist) + 0.5) / (len(plist) + 0.5)) for term, plist in self.postings.items()
        }
        return self

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        """BM25 score of every document for the query."""
        scores = np.zeros(len(self))
        if not len(self) or self.avgdl == 0:
            return scores
        norm = self.k1 * (1.0 - self.b + self.b * self.doc_lengths / self.avgdl)
        for term in set(query_tokens):
            for doc_index, tf in self.postings.get(term, ()):
                scores[doc_index] += self.idf[term] * tf * (self.k1 + 1.0) / (tf + norm[doc_index])
        return scores

    def search(self, query_tokens: list[str], k: int) -> list[tuple[int, float]]:
        """Top ``k`` documents with a positive score, best first. Ties keep index order."""
        scores = self.scores(query_tokens)
        order = np.argsort(-scores, kind="stable")[:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]
