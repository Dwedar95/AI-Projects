"""End-to-end demo of the platform core on synthetic data, offline and on CPU.

Run with ``make demo`` or ``PYTHONPATH=src python examples/demo.py``. Everything
is written to a temporary directory. Lines starting with ``$ llmops`` are real
CLI invocations executed in-process.
"""

from __future__ import annotations

import tempfile
from collections import Counter
from pathlib import Path

from llmops_gate.cli import build_predictor, main
from llmops_gate.cost import BudgetGovernor, PricingTable, TokenBudget, UsageTracker
from llmops_gate.datasets import generate_tickets, intent_keywords, write_jsonl
from llmops_gate.optimize import format_report, run_benchmark
from llmops_gate.providers import KeywordMockLLM
from llmops_gate.registry import Registry, Stage, render_prompt
from llmops_gate.training import save_model, train_intent_classifier

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
DATASET = str(BENCHMARKS / "intent_benchmark.jsonl")


def llmops(registry: Path, *args: str) -> int:
    shown = " ".join(
        a.replace(str(registry.parent), "<tmp>").replace(str(BENCHMARKS), "benchmarks")
        for a in args
    )
    print(f"$ llmops {shown}")
    code = main(["--registry", str(registry), *args])
    print(f"[exit code {code}]\n")
    return code


def prompt_template(descriptions: dict[str, str], extra: str = "") -> str:
    labels = "\n".join(f"- {label}: {text}" for label, text in descriptions.items())
    return (
        "Classify the support ticket into exactly one label.\n"
        f"Labels:\n{labels}\n{extra}Answer with the label only.\nTicket: {{ticket}}\n"
    )


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def model_lifecycle(tmp: Path, registry: Path) -> None:
    section("1. Model registry and evaluation gate (ONNX intent classifier, synthetic tickets)")
    candidates = [
        ("v1: first release, 2000 training tickets", 2000, (64,)),
        ("v2: trained on a truncated export, 60 tickets", 60, (64,)),
        ("v3: retrained on 4000 tickets with a wider layer", 4000, (128,)),
    ]
    for title, train_size, hidden in candidates:
        print(f"--- {title}")
        records = generate_tickets(train_size, seed=1, id_prefix="train")
        onnx_bytes, metadata = train_intent_classifier(records, hidden=hidden)
        artifact = save_model(tmp / "artifacts" / "intent.onnx", onnx_bytes, metadata)
        llmops(
            registry,
            "register",
            "model",
            "--name",
            "intent-classifier",
            "--artifact",
            str(artifact),
        )
        llmops(
            registry,
            "evaluate",
            "model",
            "--name",
            "intent-classifier",
            "--dataset",
            DATASET,
            "--gates",
            str(BENCHMARKS / "gates.yaml"),
        )
        llmops(registry, "promote", "model", "--name", "intent-classifier", "--to", "staging")
        if Registry(registry).get("model", "intent-classifier").stage is Stage.STAGING:
            llmops(
                registry, "promote", "model", "--name", "intent-classifier", "--to", "production"
            )
    llmops(registry, "list")


def prompt_lifecycle(tmp: Path, registry: Path) -> None:
    section("2. Prompt registry and evaluation gate (mock keyword LLM, simulated latency)")
    keywords = intent_keywords()
    versions = [
        ("v1: every label described", {k: ", ".join(v) for k, v in keywords.items()}, ""),
        (
            "v2: descriptions cut to save tokens",
            {k: ", ".join(v[:5]) for k, v in keywords.items()},
            "",
        ),
        (
            "v3: v1 plus a tie-breaking instruction",
            {k: ", ".join(v) for k, v in keywords.items()},
            "If several labels fit, choose the one with the most matching terms.\n",
        ),
    ]
    for title, descriptions, extra in versions:
        print(f"--- {title}")
        path = tmp / "prompt.txt"
        path.write_text(prompt_template(descriptions, extra), encoding="utf-8")
        llmops(
            registry, "register", "prompt", "--name", "intent-prompt", "--template-file", str(path)
        )
        llmops(
            registry,
            "evaluate",
            "prompt",
            "--name",
            "intent-prompt",
            "--dataset",
            DATASET,
            "--gates",
            str(BENCHMARKS / "prompt_gates.yaml"),
        )
        llmops(registry, "promote", "prompt", "--name", "intent-prompt", "--to", "staging")
        if Registry(registry).get("prompt", "intent-prompt").stage is Stage.STAGING:
            llmops(registry, "promote", "prompt", "--name", "intent-prompt", "--to", "production")
    llmops(registry, "list")


def drift_monitoring(tmp: Path, registry: Path) -> None:
    section("3. Drift monitoring on inference logs of the production model")
    store = Registry(registry)
    production = store.in_stage("model", "intent-classifier", Stage.PRODUCTION)
    predictor = build_predictor(store, production)
    windows = {
        "reference": generate_tickets(600, seed=7),
        "current_stable": generate_tickets(600, seed=8),
        "current_shifted": generate_tickets(
            600, seed=9, keyword_noise=0.45, priors=[0.1, 0.1, 0.6, 0.1, 0.1], new_term_rate=0.5
        ),
    }
    for name, tickets in windows.items():
        predictions = predictor.predict([t["text"] for t in tickets])
        logs = [
            {"text": t["text"], "label": p.label, "score": p.score, "latency_ms": p.latency_ms}
            for t, p in zip(tickets, predictions, strict=True)
        ]
        write_jsonl(tmp / f"{name}.jsonl", logs)
    print(f"logs produced by {production.ref}; the shifted window has a different class mix,")
    print("more keyword noise and vocabulary that was never seen in training\n")
    for name in ("current_stable", "current_shifted"):
        llmops(
            registry,
            "monitor",
            "--reference",
            str(tmp / "reference.jsonl"),
            "--current",
            str(tmp / f"{name}.jsonl"),
            "--rules",
            str(BENCHMARKS / "alert_rules.yaml"),
        )


def cost_governance(registry: Path) -> None:
    section("4. Cost and latency tracking with token budget governance")
    print("prices are illustrative example values, latency comes from the mock's linear model\n")
    pricing = PricingTable.example()
    tracker = UsageTracker(pricing)
    budgets = {
        "support-bot": TokenBudget(monthly_tokens=60_000, monthly_usd=0.25),
        "analytics": TokenBudget(monthly_tokens=2_000_000, monthly_usd=20.0),
    }
    governor = BudgetGovernor(tracker, budgets, fallback_model="small-llm")
    template = Registry(registry).in_stage("prompt", "intent-prompt", Stage.PRODUCTION).template
    provider = KeywordMockLLM()
    tickets = generate_tickets(500, seed=11)
    decisions: dict[str, Counter] = {team: Counter() for team in budgets}
    now = 1_790_000_000.0  # fixed timestamp so the demo is reproducible
    for i, ticket in enumerate(tickets):
        team = "support-bot" if i % 5 else "analytics"
        prompt = render_prompt(template, ticket=ticket["text"])
        decision = governor.check(team, "large-llm", len(prompt) // 4, now=now)
        decisions[team][decision.action.value] += 1
        if decision.model is None:
            continue
        completion = provider.complete(prompt, decision.model)
        tracker.record(
            team,
            decision.model,
            completion.input_tokens,
            completion.output_tokens,
            completion.latency_ms,
            ts=now,
        )
    print(f"{'team':<14}{'allow':>7}{'downgrade':>11}{'deny':>6}")
    for team, counts in decisions.items():
        print(f"{team:<14}{counts['allow']:>7}{counts['downgrade']:>11}{counts['deny']:>6}")
    for group in ("team", "model"):
        print(
            f"\n{group:<14}{'requests':>9}{'in tokens':>11}{'cost USD':>10}{'USD/1k req':>12}"
            f"{'p50 ms':>9}{'p95 ms':>9}"
        )
        for row in tracker.summary(group):
            print(
                f"{row[group]:<14}{row['requests']:>9}{row['input_tokens']:>11}"
                f"{row['cost_usd']:>10.4f}{row['cost_per_1k_requests_usd']:>12.4f}"
                f"{row['latency_p50_ms']:>9.1f}{row['latency_p95_ms']:>9.1f}"
            )
    for team, budget in budgets.items():
        tokens, usd = tracker.month_totals(team, "2026-09")
        print(
            f"\n{team}: {tokens} of {budget.monthly_tokens} tokens, "
            f"{usd:.4f} of {budget.monthly_usd:.2f} USD used",
            end="",
        )
    print()


def optimisation_benchmark() -> None:
    section("5. Model optimisation benchmark (real CPU measurements)")
    print(format_report(run_benchmark()))


def main_demo() -> None:
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        registry = tmp / "registry"
        model_lifecycle(tmp, registry)
        prompt_lifecycle(tmp, registry)
        drift_monitoring(tmp, registry)
        cost_governance(registry)
    optimisation_benchmark()


if __name__ == "__main__":
    main_demo()
