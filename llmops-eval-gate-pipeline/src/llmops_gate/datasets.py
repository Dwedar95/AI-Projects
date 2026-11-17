"""JSONL benchmark datasets and a synthetic support-ticket generator.

All data produced here is SYNTHETIC. Tickets are sampled from small per-intent
vocabularies with controlled keyword noise, which gives a classification task
that is neither trivial nor dependent on any real customer data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

INTENTS: list[str] = ["billing", "shipping", "technical", "account", "returns"]

_KEYWORDS: dict[str, list[str]] = {
    "billing": [
        "invoice", "charge", "payment", "billing", "card", "price",
        "subscription", "receipt", "tax", "overcharged",
    ],
    "shipping": [
        "delivery", "shipping", "tracking", "parcel", "courier", "delayed",
        "address", "dispatch", "arrive", "package",
    ],
    "technical": [
        "error", "crash", "bug", "install", "update", "login",
        "screen", "slow", "freeze", "driver",
    ],
    "account": [
        "password", "account", "email", "profile", "username", "verification",
        "locked", "settings", "security", "access",
    ],
    "returns": [
        "return", "refund", "exchange", "warranty", "damaged", "replacement",
        "broken", "defective", "label", "policy",
    ],
}  # fmt: skip

_FILLER: list[str] = [
    "hello", "please", "help", "my", "the", "with", "about", "need", "have", "issue",
    "thanks", "order", "yesterday", "still", "again", "team", "support", "question", "is", "not",
]  # fmt: skip

# Vocabulary that only appears after the simulated product launch (used for drift tests).
_NEW_PRODUCT_TERMS: list[str] = [
    "api", "webhook", "sdk", "token", "endpoint", "quota", "latency", "integration",
]  # fmt: skip


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping blank lines."""
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    """Write records as JSONL with stable key order."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def generate_tickets(
    n: int,
    seed: int,
    keyword_noise: float = 0.3,
    priors: list[float] | None = None,
    new_term_rate: float = 0.0,
    id_prefix: str = "t",
) -> list[dict[str, Any]]:
    """Generate ``n`` synthetic labelled support tickets.

    Args:
        n: number of tickets.
        seed: RNG seed, the output is fully determined by the arguments.
        keyword_noise: probability that an intent keyword is replaced by a keyword
            of a different intent. Higher values make the task harder.
        priors: class probabilities in ``INTENTS`` order, uniform when omitted.
        new_term_rate: probability that a filler word is replaced by vocabulary
            that does not occur in the training distribution (covariate shift).
        id_prefix: prefix for record ids.
    """
    if not 0.0 <= keyword_noise <= 1.0 or not 0.0 <= new_term_rate <= 1.0:
        raise ValueError("rates must be in [0, 1]")
    rng = np.random.default_rng(seed)
    probabilities = np.asarray(priors if priors is not None else [1 / len(INTENTS)] * len(INTENTS))
    probabilities = probabilities / probabilities.sum()
    records = []
    for i in range(n):
        label = INTENTS[int(rng.choice(len(INTENTS), p=probabilities))]
        words: list[str] = []
        for _ in range(int(rng.integers(3, 6))):
            source = label
            if rng.random() < keyword_noise:
                source = str(rng.choice([x for x in INTENTS if x != label]))
            words.append(str(rng.choice(_KEYWORDS[source])))
        for _ in range(int(rng.integers(4, 9))):
            if rng.random() < new_term_rate:
                words.append(str(rng.choice(_NEW_PRODUCT_TERMS)))
            else:
                words.append(str(rng.choice(_FILLER)))
        rng.shuffle(words)
        records.append({"id": f"{id_prefix}-{i:05d}", "text": " ".join(words), "label": label})
    return records


def intent_keywords() -> dict[str, list[str]]:
    """Copy of the per-intent vocabularies (used to write prompt label descriptions)."""
    return {k: list(v) for k, v in _KEYWORDS.items()}
