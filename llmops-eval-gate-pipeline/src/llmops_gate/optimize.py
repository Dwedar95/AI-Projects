"""Model optimisation benchmark: float32 ONNX versus dynamically quantised int8 ONNX.

Everything here really runs on CPU. The model is an MLP trained on the
scikit-learn ``digits`` data set (real data that ships with scikit-learn, 1797
8x8 images). It is exported with skl2onnx, quantised with onnxruntime's dynamic
quantisation (weights to int8, ``MatMul`` becomes ``MatMulInteger``) and each
variant is measured for accuracy, file size and latency.
"""

from __future__ import annotations

import logging
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from pydantic import BaseModel
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from .training import export_onnx


class VariantResult(BaseModel):
    name: str
    accuracy: float
    agreement_with_float32: float
    size_kb: float | None
    single_p50_ms: float
    single_p95_ms: float
    batch_ms: float
    batch_rows_per_s: float


class BenchmarkReport(BaseModel):
    dataset: str
    n_train: int
    n_test: int
    hidden: list[int]
    threads: int
    variants: list[VariantResult]


def quantize_int8(float_model: Path, target: Path) -> Path:
    """Dynamic int8 weight quantisation of an ONNX model."""
    previous = logging.root.level
    logging.root.setLevel(logging.ERROR)  # silence the pre-processing hint
    try:
        quantize_dynamic(str(float_model), str(target), weight_type=QuantType.QInt8)
    finally:
        logging.root.setLevel(previous)
    return target


def time_calls(func: Callable[[], object], repeats: int, warmup: int = 20) -> np.ndarray:
    """Wall-clock duration of ``repeats`` calls in milliseconds, after ``warmup`` calls."""
    for _ in range(warmup):
        func()
    durations = np.empty(repeats)
    for i in range(repeats):
        start = time.perf_counter()
        func()
        durations[i] = (time.perf_counter() - start) * 1000.0
    return durations


def _session(path: Path, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def run_benchmark(
    hidden: tuple[int, ...] = (256, 128),
    seed: int = 0,
    repeats: int = 500,
    threads: int = 1,
    workdir: str | Path | None = None,
) -> BenchmarkReport:
    """Train, export, quantise and measure. Returns one row per variant."""
    features, targets = load_digits(return_X_y=True)
    features = (features / 16.0).astype(np.float32)
    x_train, x_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.3, random_state=seed, stratify=targets
    )
    model = MLPClassifier(hidden_layer_sizes=hidden, max_iter=500, random_state=seed)
    model.fit(x_train, y_train)

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(workdir) if workdir is not None else Path(tmp)
        directory.mkdir(parents=True, exist_ok=True)
        float_path = directory / "digits_mlp_float32.onnx"
        float_path.write_bytes(export_onnx(model, x_train.shape[1]))
        int8_path = quantize_int8(float_path, directory / "digits_mlp_int8.onnx")

        single = x_test[:1]

        def measure(
            name: str, predict: Callable[[np.ndarray], np.ndarray], size_bytes: int | None
        ) -> tuple[VariantResult, np.ndarray]:
            labels = predict(x_test)
            single_ms = time_calls(lambda: predict(single), repeats)
            batch_ms = float(np.median(time_calls(lambda: predict(x_test), 30, warmup=3)))
            result = VariantResult(
                name=name,
                accuracy=round(float((labels == y_test).mean()), 4),
                agreement_with_float32=0.0,
                size_kb=round(size_bytes / 1024.0, 1) if size_bytes is not None else None,
                single_p50_ms=round(float(np.percentile(single_ms, 50)), 4),
                single_p95_ms=round(float(np.percentile(single_ms, 95)), 4),
                batch_ms=round(batch_ms, 4),
                batch_rows_per_s=round(len(x_test) / (batch_ms / 1000.0), 0),
            )
            return result, labels

        measured = [measure("sklearn-mlp", model.predict, None)]
        for name, path in (("onnx-float32", float_path), ("onnx-int8-dynamic", int8_path)):
            session = _session(path, threads)
            input_name = session.get_inputs()[0].name

            def predict(x: np.ndarray, s=session, n=input_name) -> np.ndarray:
                return s.run(["label"], {n: x})[0]

            measured.append(measure(name, predict, path.stat().st_size))

    reference = measured[1][1]
    variants = []
    for result, labels in measured:
        result.agreement_with_float32 = round(float((labels == reference).mean()), 4)
        variants.append(result)

    return BenchmarkReport(
        dataset="scikit-learn digits (1797 samples, 64 features, 10 classes)",
        n_train=len(x_train),
        n_test=len(x_test),
        hidden=list(hidden),
        threads=threads,
        variants=variants,
    )


def format_report(report: BenchmarkReport) -> str:
    """Fixed-width table for terminals and READMEs."""
    lines = [
        f"dataset: {report.dataset}; train {report.n_train}, test {report.n_test}; "
        f"MLP hidden {report.hidden}; onnxruntime CPU, {report.threads} thread(s)",
        f"{'variant':<20}{'accuracy':>9}{'agree':>8}{'size kB':>9}"
        f"{'p50 ms':>9}{'p95 ms':>9}{'batch ms':>10}{'rows/s':>10}",
    ]
    for v in report.variants:
        size = f"{v.size_kb:.1f}" if v.size_kb is not None else "n/a"
        lines.append(
            f"{v.name:<20}{v.accuracy:>9.4f}{v.agreement_with_float32:>8.4f}{size:>9}"
            f"{v.single_p50_ms:>9.4f}{v.single_p95_ms:>9.4f}{v.batch_ms:>10.3f}"
            f"{v.batch_rows_per_s:>10.0f}"
        )
    return "\n".join(lines)
