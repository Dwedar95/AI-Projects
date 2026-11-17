"""Embedding interface with two offline implementations and an optional Azure adapter.

* :class:`HashingEmbedder`: signed feature hashing of words, word bigrams and
  character n-grams with optional IDF weighting. No vocabulary, fully deterministic.
* :class:`TfidfSvdEmbedder`: TF-IDF followed by truncated SVD (latent semantic
  analysis). Captures co-occurrence structure of the indexed corpus.
* :class:`AzureOpenAIEmbedder`: thin REST adapter. NOT exercised by the test
  suite because it needs network access and credentials.

These offline embedders are lexical-statistical. They are stand-ins that keep
the pipeline runnable on CPU without a model download, not a substitute for a
neural embedding model.
"""

from __future__ import annotations

import json
import math
import os
import urllib.request
import zlib
from collections import Counter
from typing import Protocol

import numpy as np

from rag_eval_kit.text import tokenize


class Embedder(Protocol):
    """Maps texts to L2-normalised dense vectors."""

    def fit(self, texts: list[str]) -> Embedder: ...

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class HashingEmbedder:
    """Feature hashing embedder with the signed hashing trick."""

    def __init__(self, dim: int = 512, char_ngrams: tuple[int, ...] = (3, 4), char_weight: float = 0.4) -> None:
        self.dim = dim
        self.char_ngrams = char_ngrams
        self.char_weight = char_weight
        self.idf: dict[str, float] = {}
        self.default_idf = 1.0

    def fit(self, texts: list[str]) -> HashingEmbedder:
        """Learn IDF weights for word tokens. Unseen words get the maximum IDF."""
        df: Counter[str] = Counter()
        for text in texts:
            df.update(set(tokenize(text)))
        n = len(texts)
        self.idf = {term: math.log(1.0 + n / count) for term, count in df.items()}
        self.default_idf = math.log(1.0 + n) if n else 1.0
        return self

    def _features(self, text: str) -> Counter[str]:
        tokens = tokenize(text)
        features: Counter[str] = Counter()
        for token in tokens:
            weight = self.idf.get(token, self.default_idf)
            features[f"w:{token}"] += weight
            padded = f"<{token}>"
            for n in self.char_ngrams:
                for i in range(len(padded) - n + 1):
                    features[f"c:{padded[i : i + n]}"] += self.char_weight * weight / max(len(padded) - n + 1, 1)
        for first, second in zip(tokens, tokens[1:], strict=False):
            features[f"b:{first}_{second}"] += 0.5
        return features

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim))
        for row, text in enumerate(texts):
            for feature, weight in self._features(text).items():
                digest = zlib.crc32(feature.encode("utf-8"))
                sign = 1.0 if digest & 0x80000000 else -1.0
                matrix[row, digest % self.dim] += sign * weight
        return _normalize(matrix)


class TfidfSvdEmbedder:
    """Latent semantic analysis: TF-IDF projected onto the top singular vectors."""

    def __init__(self, dim: int = 128, seed: int = 0) -> None:
        self.dim = dim
        self.seed = seed
        self._vectorizer = None
        self._svd = None

    def fit(self, texts: list[str]) -> TfidfSvdEmbedder:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, sublinear_tf=True)
        tfidf = self._vectorizer.fit_transform(texts)
        components = max(1, min(self.dim, tfidf.shape[0] - 1, tfidf.shape[1] - 1))
        self._svd = TruncatedSVD(n_components=components, random_state=self.seed)
        self._svd.fit(tfidf)
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._vectorizer is None or self._svd is None:
            raise RuntimeError("TfidfSvdEmbedder must be fitted before embedding")
        return _normalize(self._svd.transform(self._vectorizer.transform(texts)))


class AzureOpenAIEmbedder:
    """Azure OpenAI embeddings over REST. UNTESTED: needs network access and credentials.

    Reads ``AZURE_OPENAI_ENDPOINT``, ``AZURE_OPENAI_API_KEY``,
    ``AZURE_OPENAI_EMBEDDING_DEPLOYMENT`` and optionally ``AZURE_OPENAI_API_VERSION``.
    """

    def __init__(self, batch_size: int = 16, timeout: float = 30.0) -> None:
        self.endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
        self.api_key = os.environ["AZURE_OPENAI_API_KEY"]
        self.deployment = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
        self.api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01")
        self.batch_size = batch_size
        self.timeout = timeout

    def fit(self, texts: list[str]) -> AzureOpenAIEmbedder:
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        url = f"{self.endpoint}/openai/deployments/{self.deployment}/embeddings?api-version={self.api_version}"
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            body = json.dumps({"input": texts[start : start + self.batch_size]}).encode("utf-8")
            headers = {"Content-Type": "application/json", "api-key": self.api_key}
            request = urllib.request.Request(url, data=body, headers=headers)  # noqa: S310
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                payload = json.loads(response.read())
            vectors.extend(item["embedding"] for item in sorted(payload["data"], key=lambda item: item["index"]))
        return _normalize(np.array(vectors, dtype=float))


def get_embedder(name: str) -> Embedder:
    """Return an embedder by name: ``tfidf-svd``, ``hashing`` or ``azure-openai``."""
    if name == "tfidf-svd":
        return TfidfSvdEmbedder()
    if name == "hashing":
        return HashingEmbedder()
    if name == "azure-openai":
        return AzureOpenAIEmbedder()
    raise ValueError(f"unknown embedder '{name}', choose from ['azure-openai', 'hashing', 'tfidf-svd']")
