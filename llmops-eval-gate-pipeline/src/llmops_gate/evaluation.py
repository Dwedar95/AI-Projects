"""Benchmark evaluation of release candidates (ONNX models and prompt plus LLM pairs)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import onnxruntime as ort
from pydantic import BaseModel, Field

from .cost import PricingTable
from .embedding import HashingEmbedder, estimate_tokens
from .providers import LLMProvider
from .registry import render_prompt


class Prediction(BaseModel):
    """Model output for one input plus the measurements needed for the gates."""

    label: str
    score: float
    latency_ms: float
    input_tokens: int
    output_tokens: int


class Predictor(Protocol):
    model_name: str
    latency_simulated: bool

    def predict(self, texts: list[str]) -> list[Prediction]: ...


class OnnxClassifierPredictor:
    """Runs the exported intent classifier with onnxruntime on CPU.

    Latency is real wall-clock time per request (embedding plus inference,
    batch size one, single intra-op thread), measured after a warm-up call.
    """

    latency_simulated = False

    def __init__(
        self,
        model_path: str | Path,
        labels: list[str],
        embed_dim: int,
        model_name: str = "onnx-classifier",
    ) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.labels = labels
        self.embedder = HashingEmbedder(dim=embed_dim)
        self.model_name = model_name
        self._run(np.zeros((1, embed_dim), dtype=np.float32))  # warm-up

    def _run(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        label, probabilities = self.session.run(None, {self.input_name: features})
        return label, probabilities

    def predict(self, texts: list[str]) -> list[Prediction]:
        predictions = []
        for text in texts:
            start = time.perf_counter()
            features = self.embedder.embed_one(text)[None, :]
            label, probabilities = self._run(features)
            latency_ms = (time.perf_counter() - start) * 1000.0
            predictions.append(
                Prediction(
                    label=self.labels[int(label[0])],
                    score=float(probabilities[0].max()),
                    latency_ms=latency_ms,
                    input_tokens=estimate_tokens(text),
                    output_tokens=0,
                )
            )
        return predictions


class PromptLLMPredictor:
    """Classifies by rendering a registered prompt template and calling an LLM provider."""

    def __init__(
        self,
        template: str,
        provider: LLMProvider,
        model_name: str,
        labels: list[str],
        variable: str = "ticket",
    ) -> None:
        self.template = template
        self.provider = provider
        self.model_name = model_name
        self.labels = labels
        self.variable = variable
        self.latency_simulated = False

    def predict(self, texts: list[str]) -> list[Prediction]:
        predictions = []
        for text in texts:
            prompt = render_prompt(self.template, **{self.variable: text})
            completion = self.provider.complete(prompt, self.model_name)
            self.latency_simulated = self.latency_simulated or completion.latency_simulated
            answer = completion.text.strip().lower().strip(".")
            known = answer in self.labels
            predictions.append(
                Prediction(
                    label=answer if known else "unknown",
                    score=1.0 if known else 0.0,
                    latency_ms=completion.latency_ms,
                    input_tokens=completion.input_tokens,
                    output_tokens=completion.output_tokens,
                )
            )
        return predictions


class EvalResult(BaseModel):
    """Metrics of one candidate on one benchmark."""

    model_name: str
    n_examples: int
    metrics: dict[str, float]
    per_class_f1: dict[str, float]
    latency_simulated: bool
    errors: list[dict[str, Any]] = Field(default_factory=list)


def macro_f1(truth: list[str], predicted: list[str], labels: list[str]) -> dict[str, float]:
    """Per-class F1 plus the unweighted mean under the key ``macro``."""
    scores: dict[str, float] = {}
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(truth, predicted, strict=True))
        fp = sum(t != label and p == label for t, p in zip(truth, predicted, strict=True))
        fn = sum(t == label and p != label for t, p in zip(truth, predicted, strict=True))
        denominator = 2 * tp + fp + fn
        scores[label] = 2 * tp / denominator if denominator else 0.0
    scores["macro"] = float(np.mean([scores[label] for label in labels]))
    return scores


def evaluate(
    predictor: Predictor,
    records: list[dict[str, Any]],
    pricing: PricingTable,
    max_errors: int = 5,
) -> EvalResult:
    """Run ``predictor`` over benchmark ``records`` (``text`` and ``label`` keys)."""
    if not records:
        raise ValueError("benchmark is empty")
    truth = [r["label"] for r in records]
    predictions = predictor.predict([r["text"] for r in records])
    predicted = [p.label for p in predictions]
    labels = sorted(set(truth))
    f1 = macro_f1(truth, predicted, labels)
    latencies = np.asarray([p.latency_ms for p in predictions])
    costs = [
        pricing.cost(predictor.model_name, p.input_tokens, p.output_tokens) for p in predictions
    ]
    wrong = [
        {"id": r.get("id"), "text": r["text"], "expected": t, "predicted": p}
        for r, t, p in zip(records, truth, predicted, strict=True)
        if t != p
    ]
    metrics = {
        "accuracy": round(1.0 - len(wrong) / len(records), 4),
        "macro_f1": round(f1.pop("macro"), 4),
        "latency_p50_ms": round(float(np.percentile(latencies, 50)), 4),
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 4),
        "mean_input_tokens": round(float(np.mean([p.input_tokens for p in predictions])), 2),
        "cost_per_1k_requests_usd": round(float(np.mean(costs)) * 1000.0, 6),
    }
    return EvalResult(
        model_name=predictor.model_name,
        n_examples=len(records),
        metrics=metrics,
        per_class_f1={k: round(v, 4) for k, v in f1.items()},
        latency_simulated=predictor.latency_simulated,
        errors=wrong[:max_errors],
    )
