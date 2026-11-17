from pathlib import Path

import pytest

from agentic_orchestrator import MockLLM
from agentic_orchestrator.evals import (
    EvalTask,
    GateThresholds,
    LLMJudge,
    load_suite,
    regression_gate,
    run_suite,
)
from agentic_orchestrator.evals.__main__ import main, make_factory
from agentic_orchestrator.evals.harness import percentile
from agentic_orchestrator.llm import ScriptedLLM

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "evals" / "support_ops.jsonl"
BASELINE = ROOT / "evals" / "baseline.json"


@pytest.fixture(scope="module")
def tasks():
    return load_suite(SUITE)


@pytest.fixture(scope="module")
def full_report(tasks):
    return run_suite(tasks, make_factory(), judge=LLMJudge(MockLLM()))


def test_suite_loads_and_covers_all_categories(tasks):
    assert len(tasks) == 18
    assert {t.category for t in tasks} >= {"prompt_injection", "high_risk_action", "multi_step"}


def test_duplicate_task_ids_are_rejected(tmp_path):
    line = '{"id": "a", "category": "c", "goal": "What is 1 + 1?"}\n'
    path = tmp_path / "suite.jsonl"
    path.write_text(line * 2)
    with pytest.raises(ValueError, match="unique"):
        load_suite(path)


def test_full_configuration_is_accurate_and_safe(full_report):
    assert full_report.accuracy == 1.0 and full_report.safety == 1.0
    assert full_report.escalation_recall == 1.0 and full_report.over_escalations == 0
    assert full_report.latency_p95_ms >= full_report.latency_p50_ms > 0
    assert full_report.by_category["prompt_injection"] == 1.0


def test_removing_guardrails_and_gate_is_measurably_unsafe(tasks):
    report = run_suite(tasks, make_factory(guardrails=False, gate=False))
    assert report.safety < 0.7 and report.escalation_recall == 0.0
    unsafe = {r.task_id for r in report.results if not r.safe}
    assert {"inject-01", "delete-02", "email-02"} <= unsafe


def test_gate_only_configuration_stays_safe_but_over_escalates(tasks):
    report = run_suite(tasks, make_factory(guardrails=False, gate=True))
    assert report.safety == 1.0 and report.over_escalations == 1


def test_failed_expectation_is_reported_with_reason():
    task = EvalTask(id="x", category="c", goal="What is 2 + 2?", answer_contains=["5"])
    report = run_suite([task], make_factory())
    assert report.accuracy == 0.0 and report.results[0].failures == ["answer lacks '5'"]


def test_regression_gate_passes_against_committed_baseline(full_report):
    import json

    outcome = regression_gate(full_report, json.loads(BASELINE.read_text()))
    assert outcome.passed, outcome.failures


def test_regression_gate_blocks_accuracy_safety_and_latency_regressions(full_report):
    baseline = {"accuracy": 1.0, "safety": 1.0, "over_escalations": 0}
    worse = full_report.model_copy(
        update={"accuracy": 0.9, "safety": 0.95, "latency_p95_ms": 900.0, "over_escalations": 2}
    )
    outcome = regression_gate(worse, baseline, GateThresholds(max_accuracy_drop=0.05))
    assert not outcome.passed
    text = " ".join(outcome.failures)
    assert "accuracy" in text and "safety" in text and "latency" in text and "over-esc" in text


def test_gate_tolerance_allows_small_accuracy_drop(full_report):
    slightly = full_report.model_copy(update={"accuracy": 0.96})
    baseline = {"accuracy": 1.0, "safety": 1.0}
    assert regression_gate(slightly, baseline, GateThresholds(max_accuracy_drop=0.05)).passed
    assert not regression_gate(slightly, baseline).passed


def test_cli_exit_codes_reflect_the_gate(capsys):
    assert main(["--suite", str(SUITE), "--baseline", str(BASELINE)]) == 0
    assert main(["--suite", str(SUITE), "--baseline", str(BASELINE), "--no-gate"]) == 1
    assert "REGRESSION GATE: FAILED" in capsys.readouterr().out


def test_judge_scores_and_survives_invalid_judge_output():
    judge = LLMJudge(MockLLM())
    assert judge.grade("q", "ships in 3 days", "It ships in 3 days.").score == 1.0
    assert judge.grade("q", "ships in 3 days", "unrelated").score == 0.0
    broken = LLMJudge(ScriptedLLM(["bad", "bad"])).grade("q", "ref", "ans")
    assert broken.score == 0.0 and "invalid" in broken.rationale


def test_percentile_interpolates():
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert percentile([5.0], 95) == 5.0 and percentile([], 50) == 0.0
