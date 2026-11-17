"""Evaluation framework: retrieval, answer, safety and latency metrics with a gate."""

from rag_eval_kit.evaluation.harness import (
    EvalReport,
    QAExample,
    load_probes,
    load_qa,
    run_evaluation,
)
from rag_eval_kit.evaluation.report import GateResult, apply_gate, load_thresholds, render_markdown

__all__ = [
    "EvalReport",
    "GateResult",
    "QAExample",
    "apply_gate",
    "load_probes",
    "load_qa",
    "load_thresholds",
    "render_markdown",
    "run_evaluation",
]
