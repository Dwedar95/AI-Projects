"""Metric definitions. All functions are pure and work on ids and plain values."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence

from rag_eval_kit.generation import Answer
from rag_eval_kit.retrieval import SearchHit
from rag_eval_kit.text import tokenize


def dedupe(ranked: Iterable[str]) -> list[str]:
    """Remove repeated ids, keeping the first (best ranked) occurrence."""
    return list(dict.fromkeys(ranked))


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Share of relevant ids found in the top ``k`` of the ranking."""
    if not relevant:
        raise ValueError("recall is undefined without relevant items")
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant id, 0 when none is retrieved."""
    return next((1.0 / rank for rank, item in enumerate(ranked, start=1) if item in relevant), 0.0)


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Normalised discounted cumulative gain with binary relevance."""
    if not relevant:
        raise ValueError("nDCG is undefined without relevant items")
    dcg = sum(1.0 / math.log2(rank + 1) for rank, item in enumerate(ranked[:k], start=1) if item in relevant)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


def percentile(values: Sequence[float], q: float) -> float:
    """Percentile with linear interpolation between closest ranks, ``q`` in [0, 100]."""
    if not values:
        raise ValueError("percentile of an empty sequence")
    if not 0 <= q <= 100:
        raise ValueError("q must be between 0 and 100")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100.0
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sentence_support(sentence: str, evidence: str) -> float:
    """Share of the sentence's content tokens that occur in the evidence text."""
    tokens = tokenize(sentence)
    if not tokens:
        return 1.0
    available = Counter(tokenize(evidence))
    return sum(1 for token in tokens if available[token] > 0) / len(tokens)


def groundedness(answer: Answer, hits: Sequence[SearchHit], threshold: float = 0.8) -> float:
    """Share of answer sentences supported by at least one chunk they cite.

    A sentence without a citation counts as unsupported. Support is lexical
    (token containment), so it detects unsupported additions and wrong citations
    but not subtle contradictions.
    """
    if not answer.sentences:
        return 0.0
    chunks = {
        c.marker: next((h.chunk.text for h in hits if h.chunk.chunk_id == c.chunk_id), "") for c in answer.citations
    }
    supported = 0
    for sentence in answer.sentences:
        best = max((sentence_support(sentence.text, chunks.get(m, "")) for m in sentence.markers), default=0.0)
        supported += best >= threshold
    return supported / len(answer.sentences)


def citation_precision(answer: Answer, relevant_docs: set[str]) -> float | None:
    """Share of cited documents that are labelled relevant. None when nothing is cited."""
    cited = [c.doc_id for c in answer.citations]
    return sum(doc in relevant_docs for doc in cited) / len(cited) if cited else None


def citation_recall(answer: Answer, relevant_docs: set[str]) -> float:
    """Share of the labelled relevant documents that the answer cites."""
    return len({c.doc_id for c in answer.citations} & relevant_docs) / len(relevant_docs)


def contains_keywords(text: str, keywords: Sequence[str]) -> bool:
    """True when every expected key phrase occurs in the text, ignoring case."""
    lowered = text.lower()
    return all(keyword.lower() in lowered for keyword in keywords)


def precision_recall(true_positive: int, false_positive: int, false_negative: int) -> tuple[float, float]:
    """Precision and recall, each 1.0 when its denominator is zero."""
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 1.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 1.0
    return precision, recall
