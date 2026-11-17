"""Hybrid retrieval: BM25, dense cosine search, reciprocal rank fusion and reranking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

from rag_eval_kit.bm25 import BM25Index
from rag_eval_kit.documents import Chunk
from rag_eval_kit.embeddings import Embedder
from rag_eval_kit.text import tokenize

Mode = Literal["bm25", "dense", "hybrid"]


@dataclass(frozen=True)
class SearchHit:
    """A retrieved chunk with its final score and the per-stage ranks that produced it."""

    chunk: Chunk
    score: float
    ranks: dict[str, int] = field(default_factory=dict)


class DenseIndex:
    """Exact cosine similarity search over L2-normalised vectors."""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.matrix = np.zeros((0, 0))

    def fit(self, texts: list[str]) -> DenseIndex:
        self.embedder.fit(texts)
        self.matrix = self.embedder.embed(texts)
        return self

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        if self.matrix.shape[0] == 0:
            return []
        similarities = self.matrix @ self.embedder.embed([query])[0]
        order = np.argsort(-similarities, kind="stable")[:k]
        return [(int(i), float(similarities[i])) for i in order]


def reciprocal_rank_fusion(
    rankings: dict[str, list[int]], k: int = 60, weights: dict[str, float] | None = None
) -> list[tuple[int, float]]:
    """Fuse ranked id lists: score(d) = sum over lists of weight / (k + rank(d)), rank from 1.

    RRF uses ranks only, so BM25 scores and cosine similarities need no calibration.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    fused: dict[int, float] = {}
    for name, ranking in rankings.items():
        weight = (weights or {}).get(name, 1.0)
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + weight / (k + rank)
    return sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))


class Reranker(Protocol):
    """Reorders candidate hits for a query."""

    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]: ...


def minimal_window(positions: list[list[int]]) -> int | None:
    """Length of the shortest token span that contains one position from every list."""
    if not positions or any(not p for p in positions):
        return None
    events = sorted((pos, term) for term, plist in enumerate(positions) for pos in plist)
    need, have, counts = len(positions), 0, [0] * len(positions)
    best, left = None, 0
    for pos, term in events:
        counts[term] += 1
        have += counts[term] == 1
        while have == need:
            span = pos - events[left][0] + 1
            best = span if best is None or span < best else best
            counts[events[left][1]] -= 1
            have -= counts[events[left][1]] == 0
            left += 1
    return best


class FeatureReranker:
    """Deterministic reranker built from lexical features.

    Features per candidate: IDF-weighted query term coverage in the body,
    coverage in title and section, proximity of the matched terms (shortest
    window) and the prior rank from first-stage retrieval. It stands in for a
    cross-encoder so that the reranking stage is testable offline.
    """

    def __init__(
        self, idf: dict[str, float], weights: tuple[float, float, float, float] = (0.45, 0.2, 0.15, 0.2)
    ) -> None:
        self.idf = idf
        self.default_idf = max(idf.values(), default=1.0)
        self.weights = weights

    def _coverage(self, query_terms: list[str], tokens: set[str]) -> float:
        total = sum(self.idf.get(t, self.default_idf) for t in query_terms)
        matched = sum(self.idf.get(t, self.default_idf) for t in query_terms if t in tokens)
        return matched / total if total else 0.0

    def features(self, query: str, hit: SearchHit, prior_rank: int) -> tuple[float, float, float, float]:
        query_terms = list(dict.fromkeys(tokenize(query)))
        body = tokenize(hit.chunk.text)
        heading = set(tokenize(f"{hit.chunk.title} {hit.chunk.section}"))
        matched = [t for t in query_terms if t in set(body)]
        proximity = 0.0
        if len(matched) >= 2:
            window = minimal_window([[i for i, tok in enumerate(body) if tok == t] for t in matched])
            proximity = len(matched) / window if window else 0.0
        return (
            self._coverage(query_terms, set(body)),
            self._coverage(query_terms, heading),
            proximity,
            1.0 / prior_rank,
        )

    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        scored = []
        for rank, hit in enumerate(hits, start=1):
            score = float(np.dot(self.weights, self.features(query, hit, rank)))
            scored.append(SearchHit(hit.chunk, score, {**hit.ranks, "first_stage": rank}))
        return sorted(scored, key=lambda h: -h.score)


class HybridRetriever:
    """Indexes chunks once and answers queries in BM25, dense or hybrid mode."""

    def __init__(self, embedder: Embedder, rrf_k: int = 60, candidates: int = 20) -> None:
        self.bm25 = BM25Index()
        self.dense = DenseIndex(embedder)
        self.rrf_k = rrf_k
        self.candidates = candidates
        self.chunks: list[Chunk] = []
        self.reranker: Reranker | None = None

    def index(self, chunks: list[Chunk]) -> HybridRetriever:
        """Build the sparse and dense indexes and the default reranker."""
        self.chunks = list(chunks)
        texts = [chunk.indexed_text for chunk in self.chunks]
        self.bm25.fit([tokenize(text) for text in texts])
        self.dense.fit(texts)
        self.reranker = FeatureReranker(self.bm25.idf)
        return self

    def search(self, query: str, k: int = 5, mode: Mode = "hybrid", rerank: bool = False) -> list[SearchHit]:
        """Return the top ``k`` chunks for ``query``."""
        if mode not in ("bm25", "dense", "hybrid"):
            raise ValueError(f"unknown retrieval mode '{mode}'")
        depth = max(self.candidates, k)
        sparse = self.bm25.search(tokenize(query), depth) if mode != "dense" else []
        dense = self.dense.search(query, depth) if mode != "bm25" else []
        ranks = {
            "bm25": {i: r for r, (i, _) in enumerate(sparse, start=1)},
            "dense": {i: r for r, (i, _) in enumerate(dense, start=1)},
        }
        if mode == "bm25":
            scored = sparse
        elif mode == "dense":
            scored = dense
        else:
            scored = reciprocal_rank_fusion(
                {"bm25": [i for i, _ in sparse], "dense": [i for i, _ in dense]}, self.rrf_k
            )
        hits = [
            SearchHit(self.chunks[i], score, {name: r[i] for name, r in ranks.items() if i in r}) for i, score in scored
        ]
        if rerank and self.reranker is not None:
            hits = self.reranker.rerank(query, hits[:depth])
        return hits[:k]
