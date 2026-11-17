"""Small TF-IDF retriever over a Markdown knowledge base. No external index, numpy only."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i if in is it its my of on or our the their "
    "this to was what when which who will with you your".split()
)
_TOKEN = re.compile(r"[a-z0-9]+")


def stem(token: str) -> str:
    """Crude suffix stripper so that 'refunds', 'refunded' and 'refund' share a term."""
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def tokenize(text: str, stemming: bool = True) -> list[str]:
    """Lowercase, split on non-alphanumerics, drop stopwords, optionally stem."""
    tokens = [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]
    return [stem(t) for t in tokens] if stemming else tokens


@dataclass(frozen=True)
class Document:
    """One retrievable chunk."""

    doc_id: str
    text: str
    source: str = ""


@dataclass(frozen=True)
class Hit:
    """A retrieval result with its cosine similarity."""

    document: Document
    score: float


class TfidfRetriever:
    """Cosine similarity over L2-normalised TF-IDF vectors with sublinear term frequency."""

    def __init__(self, documents: list[Document]):
        if not documents:
            raise ValueError("the knowledge base is empty")
        self.documents = documents
        tokenised = [tokenize(d.text) for d in documents]
        vocab = sorted({t for toks in tokenised for t in toks})
        self.vocab = {t: i for i, t in enumerate(vocab)}
        n = len(documents)
        df = Counter(t for toks in tokenised for t in set(toks))
        self.idf = np.array([math.log((1 + n) / (1 + df[t])) + 1.0 for t in vocab])
        self.matrix = np.vstack([self._vector(toks) for toks in tokenised])

    def _vector(self, tokens: list[str]) -> np.ndarray:
        vec = np.zeros(len(self.vocab))
        for term, count in Counter(tokens).items():
            idx = self.vocab.get(term)
            if idx is not None:
                vec[idx] = (1.0 + math.log(count)) * self.idf[idx]
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def search(self, query: str, k: int = 3, min_score: float = 0.0) -> list[Hit]:
        """Return the ``k`` best chunks with a score above ``min_score``, best first."""
        scores = self.matrix @ self._vector(tokenize(query))
        order = np.argsort(-scores, kind="stable")[:k]
        return [Hit(self.documents[i], float(scores[i])) for i in order if scores[i] > min_score]


def load_knowledge_base(directory: str | Path) -> list[Document]:
    """Read all Markdown files of a directory and split them into paragraph chunks.

    The nearest heading is prepended to each paragraph so that a chunk keeps its topic.
    """
    docs: list[Document] = []
    for path in sorted(Path(directory).glob("*.md")):
        heading = ""
        idx = 0
        for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8")):
            block = block.strip()
            if not block:
                continue
            if block.startswith("#"):
                heading = block.lstrip("# ").strip()
                continue
            idx += 1
            text = f"{heading}: {block}" if heading else block
            docs.append(Document(f"{path.stem}#{idx}", " ".join(text.split()), path.name))
    if not docs:
        raise ValueError(f"no Markdown content found in {directory}")
    return docs
