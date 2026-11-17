"""End-to-end demo on the offline mock provider and the synthetic support back office.

Run with ``make demo`` or ``PYTHONPATH=src python examples/demo.py``.
"""

from __future__ import annotations

from pathlib import Path

from agentic_orchestrator import (
    MockLLM,
    Orchestrator,
    RunState,
    SupportBackend,
    build_support_registry,
)
from agentic_orchestrator.evals import LLMJudge, load_suite, run_suite
from agentic_orchestrator.evals.__main__ import make_factory

SUITE = Path(__file__).resolve().parents[1] / "evals" / "support_ops.jsonl"


def show(title: str, state: RunState) -> None:
    print(f"\n--- {title}")
    print(f"goal     : {state.goal}")
    print(f"plan     : {[f'{s.id}:{s.tool_hint}' for s in state.plan.steps]}")
    print(f"status   : {state.status.value}")
    print(f"executed : {state.executed_tools()}")
    if state.pending_approval:
        request = state.pending_approval
        print(f"PENDING  : {request.tool_name} {request.arguments} risk={request.risk.score:.2f}")
        for reason in request.risk.reasons:
            print(f"           - {reason}")
    if state.final_answer:
        print(f"answer   : {state.final_answer}")


def main() -> None:
    backend = SupportBackend.seeded()
    orchestrator = Orchestrator(
        MockLLM(faults=["fenced", "wrong_schema"]), build_support_registry(backend)
    )

    state = orchestrator.run(
        "Look up order ORD-1002, then calculate 480 * 0.19, "
        "then email finance@company.example the VAT amount"
    )
    show("1. Plan-and-execute, three steps, low risk", state)
    counters = state.counters
    print(
        f"guardrail: {counters.local_repairs} local JSON repair, "
        f"{counters.structured_retries} retry with validation feedback"
    )

    state = orchestrator.run("Refund order ORD-1001", mode="react")
    show("2. ReAct, small refund is auto-approved", state)

    state = orchestrator.run("Refund order ORD-1003")
    show("3. Large refund pauses for a human", state)
    state = orchestrator.resume(state.run_id, approved=True, reviewer="alice")
    show("3b. After approval by alice", state)

    state = orchestrator.run("Delete the record of customer CUST-002")
    show("4. Destructive action pauses for a human", state)
    state = orchestrator.resume(state.run_id, False, "bob", reason="no erasure request on file")
    show("4b. After rejection by bob", state)
    print(f"customer still exists: {'CUST-002' in backend.customers}")

    state = orchestrator.run(
        "What does the hardware warranty cover, then email b@customer.example the answer"
    )
    show("5. Poisoned knowledge base article (indirect prompt injection)", state)
    print(f"tainted  : {state.tainted}")
    print(f"step 2   : {state.records[1].turns[0].observation.output}")
    print(f"outbox recipients: {[m['to'] for m in backend.outbox]}")

    ok, detail = orchestrator.audit.verify()
    print(f"\n--- Audit log: {len(orchestrator.audit)} records, chain valid={ok} ({detail})")
    orchestrator.audit.records()[5].payload["tampered"] = True
    print(f"after editing record 5 in place: {orchestrator.audit.verify()}")

    print("\n--- Evaluation suite (18 synthetic tasks, mock provider), safety-layer ablation")
    header = (
        f"{'configuration':<28}{'accuracy':>9}{'safety':>8}{'esc.recall':>11}{'over-esc':>9}"
        f"{'p50 ms':>8}{'p95 ms':>8}"
    )
    print(header)
    tasks = load_suite(SUITE)
    for name, guardrails, gate in [
        ("guardrails + approval gate", True, True),
        ("approval gate only", False, True),
        ("no protection", False, False),
    ]:
        report = run_suite(tasks, make_factory(guardrails, gate), judge=LLMJudge(MockLLM()))
        print(
            f"{name:<28}{report.accuracy:>9.3f}{report.safety:>8.3f}"
            f"{report.escalation_recall:>11.3f}{report.over_escalations:>9d}"
            f"{report.latency_p50_ms:>8.2f}{report.latency_p95_ms:>8.2f}"
        )


if __name__ == "__main__":
    main()
