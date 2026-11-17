"""Train the example intent classifier and export it to ONNX.

The model is a scikit-learn MLP on hashing embeddings. It is deliberately small:
the point of this repository is the platform around the model (registry, gates,
monitoring), and a model that trains in a second keeps CI fast.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from skl2onnx import to_onnx
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

from .datasets import INTENTS
from .embedding import HashingEmbedder


def export_onnx(model: MLPClassifier, n_features: int) -> bytes:
    """Convert a fitted MLP to ONNX with tensor outputs ``label`` and ``probabilities``."""
    sample = np.zeros((1, n_features), dtype=np.float32)
    onnx_model = to_onnx(model, sample, options={id(model): {"zipmap": False}}, target_opset=17)
    return onnx_model.SerializeToString()


def train_intent_classifier(
    records: list[dict[str, Any]],
    embed_dim: int = 256,
    hidden: tuple[int, ...] = (64,),
    seed: int = 0,
    max_iter: int = 500,
) -> tuple[bytes, dict[str, Any]]:
    """Fit the classifier on labelled tickets and return ``(onnx_bytes, sidecar_metadata)``."""
    if len(records) < len(INTENTS):
        raise ValueError("need at least one record per intent")
    embedder = HashingEmbedder(dim=embed_dim)
    features = embedder.embed([r["text"] for r in records])
    targets = np.asarray([INTENTS.index(r["label"]) for r in records])
    if len(set(targets.tolist())) != len(INTENTS):
        raise ValueError("training data must contain every intent")
    model = MLPClassifier(
        hidden_layer_sizes=hidden,
        alpha=1e-2,
        early_stopping=len(records) >= 200,  # needs enough rows for a validation split
        max_iter=max_iter,
        random_state=seed,
    )
    with warnings.catch_warnings():
        # Non-convergence is recorded in the metadata instead of printed as a warning.
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(features, targets)
    metadata = {
        "labels": INTENTS,
        "embed_dim": embed_dim,
        "hidden": list(hidden),
        "seed": seed,
        "train_size": len(records),
        "epochs": int(model.n_iter_),
        "converged": bool(model.n_iter_ < max_iter),
        "train_accuracy": round(float(model.score(features, targets)), 4),
        "framework": "scikit-learn MLPClassifier exported with skl2onnx",
    }
    return export_onnx(model, embed_dim), metadata


def save_model(path: str | Path, onnx_bytes: bytes, metadata: dict[str, Any]) -> Path:
    """Write the ONNX file and a ``<name>.json`` sidecar that ``llmops register`` picks up."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(onnx_bytes)
    sidecar = target.with_name(target.name + ".json")
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return target
