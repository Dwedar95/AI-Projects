"""Evaluation harness: task suites, deterministic checks, LLM-as-judge and a regression gate."""

from .gate import GateResult, GateThresholds, regression_gate
from .harness import EvalReport, EvalTask, TaskResult, load_suite, run_suite
from .judge import Judge, JudgeVerdict, LLMJudge

__all__ = [
    "EvalReport",
    "EvalTask",
    "GateResult",
    "GateThresholds",
    "Judge",
    "JudgeVerdict",
    "LLMJudge",
    "TaskResult",
    "load_suite",
    "regression_gate",
    "run_suite",
]
