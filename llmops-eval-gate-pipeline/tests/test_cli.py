import json

import pytest

from llmops_gate.cli import main
from llmops_gate.datasets import generate_tickets, intent_keywords, write_jsonl
from llmops_gate.registry import Registry, Stage

from .conftest import BENCHMARKS, DATASET, GATES


def run(registry, *args):
    return main(["--registry", str(registry.root), *args])


def _evaluate(registry, kind="model", name="intent-classifier", gates=GATES, *extra):
    return run(
        registry, "evaluate", kind, "--name", name, "--dataset", str(DATASET),
        "--gates", str(gates), *extra,
    )  # fmt: skip


def test_register_evaluate_promote_happy_path(registry, good_model, capsys, tmp_path):
    assert run(registry, "register", "model", "--name", "intent-classifier",
               "--artifact", str(good_model), "--meta", "owner=ml-platform") == 0  # fmt: skip
    report = tmp_path / "out" / "report.json"
    assert _evaluate(registry, "model", "intent-classifier", GATES, "--report", str(report)) == 0
    assert run(registry, "promote", "model", "--name", "intent-classifier", "--to", "staging") == 0
    assert run(registry, "promote", "model", "--name", "intent-classifier",
               "--to", "production") == 0  # fmt: skip
    entry = registry.get("model", "intent-classifier")
    assert entry.stage is Stage.PRODUCTION and entry.metadata["owner"] == "ml-platform"
    assert entry.metadata["train_size"] == 2000  # picked up from the sidecar
    assert json.loads(report.read_text())["gate"]["passed"] is True
    assert run(registry, "list") == 0
    out = capsys.readouterr().out
    assert "GATE: PASSED" in out and "production" in out


def test_regressed_candidate_fails_gate_and_cannot_be_promoted(
    registry, good_model, weak_model, capsys
):
    for artifact in (good_model, weak_model):
        run(registry, "register", "model", "--name", "intent-classifier",
            "--artifact", str(artifact))  # fmt: skip
        code = _evaluate(registry)
        if artifact is good_model:
            assert code == 0
            run(registry, "promote", "model", "--name", "intent-classifier", "--to", "staging")
            run(registry, "promote", "model", "--name", "intent-classifier", "--to", "production")
    assert code == 1
    assert run(registry, "promote", "model", "--name", "intent-classifier", "--to", "staging") == 1
    out = capsys.readouterr().out
    assert "baseline: model:intent-classifier:v1" in out and "PROMOTION BLOCKED" in out
    assert registry.in_stage("model", "intent-classifier", Stage.PRODUCTION).version == 1
    assert registry.get("model", "intent-classifier", 2).stage is Stage.NONE
    # An explicit override is possible and leaves a trace.
    assert run(registry, "promote", "model", "--name", "intent-classifier", "--to", "staging",
               "--override-reason", "incident 42") == 0  # fmt: skip
    assert registry.history()[-1]["override_reason"] == "incident 42"


def test_explicit_baseline_file_is_used(registry, weak_model, good_model, capsys, tmp_path):
    run(registry, "register", "model", "--name", "intent-classifier", "--artifact", str(good_model))
    baseline = BENCHMARKS / "baseline_metrics.json"
    assert _evaluate(registry, "model", "intent-classifier", GATES,
                     "--baseline-metrics", str(baseline)) == 0  # fmt: skip
    strict = tmp_path / "strict.json"
    strict.write_text(json.dumps({"accuracy": 0.99, "macro_f1": 0.99}))
    assert _evaluate(registry, "model", "intent-classifier", GATES,
                     "--baseline-metrics", str(strict)) == 1  # fmt: skip
    assert "explicit baseline file" in capsys.readouterr().out


def test_prompt_versions_through_the_cli(registry, tmp_path, capsys):
    keywords = intent_keywords()
    gates = BENCHMARKS / "prompt_gates.yaml"

    def register(n_keywords):
        labels = "\n".join(f"- {k}: {', '.join(v[:n_keywords])}" for k, v in keywords.items())
        path = tmp_path / "prompt.txt"
        path.write_text(f"Labels:\n{labels}\nAnswer with the label only.\nTicket: {{ticket}}\n")
        return run(registry, "register", "prompt", "--name", "intent-prompt",
                   "--template-file", str(path))  # fmt: skip

    assert register(10) == 0
    assert _evaluate(registry, "prompt", "intent-prompt", gates) == 0
    run(registry, "promote", "prompt", "--name", "intent-prompt", "--to", "staging")
    run(registry, "promote", "prompt", "--name", "intent-prompt", "--to", "production")
    assert register(4) == 0
    assert _evaluate(registry, "prompt", "intent-prompt", gates) == 1
    assert "latency simulated" in capsys.readouterr().out
    assert register(4) == 2  # identical template


def test_monitor_exit_codes(tmp_path, capsys):
    def logs(name, tickets, score):
        path = tmp_path / name
        write_jsonl(
            path,
            [
                {"text": t["text"], "label": t["label"], "score": score + 0.001 * (i % 50)}
                for i, t in enumerate(tickets)
            ],
        )
        return str(path)  # fmt: skip

    rules = str(BENCHMARKS / "alert_rules.yaml")
    reference = logs("ref.jsonl", generate_tickets(200, seed=1), 0.9)
    stable = logs("stable.jsonl", generate_tickets(200, seed=2), 0.9)
    shifted = logs(
        "shift.jsonl", generate_tickets(200, seed=3, priors=[0, 0, 1, 0, 0], new_term_rate=0.8), 0.6
    )
    assert main(["monitor", "--reference", reference, "--current", stable, "--rules", rules]) == 0
    report = tmp_path / "drift.json"
    assert main(["monitor", "--reference", reference, "--current", shifted, "--rules", rules,
                 "--report", str(report)]) == 2  # fmt: skip
    assert json.loads(report.read_text())["status"] == "critical"
    assert "ALERT [critical]" in capsys.readouterr().out


@pytest.mark.parametrize(
    "args",
    [
        ["register", "model", "--name", "x"],
        ["register", "prompt", "--name", "x"],
        ["evaluate", "model", "--name", "missing", "--dataset", "d", "--gates", "g"],
        ["promote", "model", "--name", "missing", "--to", "staging"],
        ["register", "model", "--name", "x", "--artifact", "does-not-exist.onnx"],
    ],
)
def test_usage_errors_exit_with_code_two(tmp_path, args, capsys):
    assert main(["--registry", str(tmp_path / "r"), *args]) == 2
    assert capsys.readouterr().err.startswith("error:")
    assert not Registry(tmp_path / "r").names("model")
