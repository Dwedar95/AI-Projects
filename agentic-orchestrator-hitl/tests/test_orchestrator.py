import json

import pytest

from agentic_orchestrator import (
    ApprovalGate,
    GuardrailConfig,
    JsonFileStateStore,
    MockLLM,
    Orchestrator,
    RunStatus,
    ScriptedLLM,
)
from agentic_orchestrator.hitl import ApprovalError
from agentic_orchestrator.orchestrator import RunNotFoundError


def _tool(name, **arguments):
    return json.dumps({"thought": "t", "action": "tool", "tool_name": name, "arguments": arguments})


def _finish(answer):
    return json.dumps({"thought": "t", "action": "finish", "final_answer": answer})


def test_single_step_run_completes(orchestrator):
    state = orchestrator.run("What is 17 * 23 + 4?")
    assert state.status is RunStatus.COMPLETED
    assert state.final_answer == "The result of 17 * 23 + 4 is 395."
    assert state.executed_tools() == ["calculator"]
    assert state.counters.llm_calls == 3  # plan, act, finish


def test_plan_and_execute_passes_results_between_steps(orchestrator, backend):
    state = orchestrator.run(
        "Look up order ORD-1002, then calculate 480 * 0.19, "
        "then email finance@company.example the VAT amount"
    )
    assert [s.tool_hint for s in state.plan.steps] == ["lookup_order", "calculator", "send_email"]
    assert state.executed_tools() == ["lookup_order", "calculator", "send_email"]
    assert "91.2" in backend.outbox[0]["body"]
    assert state.final_answer.startswith("(1) Order ORD-1002")


def test_react_mode_skips_the_planner_and_chains_tools(orchestrator, backend):
    state = orchestrator.run("Refund order ORD-1001", mode="react")
    assert len(state.plan.steps) == 1 and state.plan.steps[0].tool_hint is None
    assert state.executed_tools() == ["lookup_order", "issue_refund"]
    assert backend.refunds == [{"order_id": "ORD-1001", "amount": 40.0}]


def test_high_risk_action_pauses_and_resumes_after_approval(orchestrator, backend):
    state = orchestrator.run("Refund order ORD-1002")
    assert state.status is RunStatus.AWAITING_APPROVAL
    assert state.pending_approval.tool_name == "issue_refund"
    assert state.pending_approval.arguments == {"order_id": "ORD-1002", "amount": 480.0}
    assert backend.refunds == []  # nothing happened yet
    assert [s.run_id for s in orchestrator.pending_approvals()] == [state.run_id]

    done = orchestrator.resume(state.run_id, approved=True, reviewer="alice")
    assert done.status is RunStatus.COMPLETED
    assert backend.refunds == [{"order_id": "ORD-1002", "amount": 480.0}]
    assert done.approvals[0].reviewer == "alice" and done.pending_approval is None
    with pytest.raises(ApprovalError):
        orchestrator.resume(state.run_id, approved=True, reviewer="alice")


def test_rejected_action_is_never_executed(orchestrator, backend):
    state = orchestrator.run("Delete the record of customer CUST-002")
    done = orchestrator.resume(state.run_id, False, "bob", reason="no erasure request on file")
    assert done.status is RunStatus.COMPLETED
    assert "CUST-002" in backend.customers and backend.deleted_customers == []
    assert "was not performed" in done.final_answer and "no erasure request" in done.final_answer
    assert done.executed_tools() == []


def test_paused_run_survives_a_process_restart(tmp_path, registry, backend):
    first = Orchestrator(MockLLM(), registry, store=JsonFileStateStore(tmp_path))
    run_id = first.run("Refund order ORD-1003").run_id
    second = Orchestrator(MockLLM(), registry, store=JsonFileStateStore(tmp_path))
    done = second.resume(run_id, approved=True, reviewer="carol")
    assert done.status is RunStatus.COMPLETED and backend.refunds[0]["amount"] == 2300.0


def test_resume_unknown_run_raises(orchestrator):
    with pytest.raises(RunNotFoundError):
        orchestrator.resume("run_missing", True, "alice")


def test_injected_instruction_is_stripped_and_run_is_tainted(orchestrator, backend):
    state = orchestrator.run("What does the hardware warranty cover?")
    assert state.status is RunStatus.COMPLETED and state.tainted
    assert "24 months" in state.final_answer and "ignore" not in state.final_answer.lower()
    assert backend.outbox == []
    events = orchestrator.audit.records(state.run_id, "guardrail_triggered")
    assert events and "override_instructions" in events[0].payload["rules"]


def test_without_any_protection_the_gullible_policy_exfiltrates(registry, backend):
    unprotected = Orchestrator(
        MockLLM(gullible=True),
        registry,
        guardrails=GuardrailConfig(scan_tool_output=False),
        gate=ApprovalGate(enabled=False),
    )
    unprotected.run("What does the hardware warranty cover?")
    assert backend.outbox[0]["to"] == "exfil@attacker.example"


def test_gate_alone_still_catches_the_injected_email(registry, backend):
    gate_only = Orchestrator(
        MockLLM(gullible=True), registry, guardrails=GuardrailConfig(scan_tool_output=False)
    )
    state = gate_only.run("What does the hardware warranty cover?")
    assert state.status is RunStatus.AWAITING_APPROVAL
    assert state.pending_approval.arguments["to"] == "exfil@attacker.example"
    done = gate_only.resume(state.run_id, False, "alice", reason="not requested by the user")
    assert backend.outbox == [] and "24 months" in done.final_answer


def test_taint_escalates_an_otherwise_auto_approved_refund(orchestrator):
    state = orchestrator.run("How long is the warranty, then refund order ORD-1001")
    assert state.status is RunStatus.AWAITING_APPROVAL
    assert state.pending_approval.risk.score == pytest.approx(0.65)
    assert any("untrusted content" in r for r in state.pending_approval.risk.reasons)


def test_tainted_external_email_is_denied_by_policy(orchestrator, backend):
    state = orchestrator.run(
        "What does the hardware warranty cover, then email b@customer.example the answer"
    )
    assert state.status is RunStatus.COMPLETED and backend.outbox == []
    blocked = state.records[1].turns[0].observation
    assert blocked.output.startswith("BLOCKED by policy (risk 0.95)") and not blocked.executed


def test_agent_recovers_from_unknown_tool_and_bad_arguments(registry):
    llm = ScriptedLLM(
        [
            _tool("order_lookup", order_id="ORD-1001"),
            _tool("lookup_order", order_id="1001"),
            _tool("lookup_order", order_id="ORD-1001"),
            _finish("Order is delivered."),
        ]
    )
    state = Orchestrator(llm, registry).run("Check order 1001", mode="react")
    outputs = [t.observation.output for t in state.records[0].turns]
    assert outputs[0].startswith("ERROR: unknown tool 'order_lookup'")
    assert outputs[1].startswith("ERROR: invalid arguments for 'lookup_order'")
    assert state.status is RunStatus.COMPLETED and state.executed_tools() == ["lookup_order"]
    # The error observation was fed back to the model.
    assert "ERROR: unknown tool" in llm.calls[1][1].content


def test_iteration_budget_stops_a_looping_agent(registry):
    llm = ScriptedLLM([_tool("calculator", expression="1+1")] * 5)
    state = Orchestrator(llm, registry, max_react_iterations=3).run("loop", mode="react")
    assert state.status is RunStatus.FAILED and "exceeded 3 iterations" in state.error


def test_unparseable_plan_fails_the_run_cleanly(registry):
    state = Orchestrator(ScriptedLLM(["??", "??", "??"]), registry).run("anything")
    assert state.status is RunStatus.FAILED and state.error.startswith("planning failed")


def test_malformed_model_output_is_repaired_and_counted(registry):
    llm = MockLLM(faults=["fenced", "truncated", "wrong_schema"])
    state = Orchestrator(llm, registry).run("What is 2 + 2?")
    assert state.status is RunStatus.COMPLETED and "4" in state.final_answer
    assert state.counters.local_repairs == 1 and state.counters.structured_retries == 2


def test_session_memory_resolves_references_across_runs(orchestrator, backend):
    orchestrator.run("Look up order ORD-1001", session_id="s1")
    state = orchestrator.run("Refund that order", session_id="s1")
    assert state.status is RunStatus.COMPLETED
    assert backend.refunds == [{"order_id": "ORD-1001", "amount": 40.0}]


def test_audit_trail_is_complete_and_verifiable(orchestrator):
    state = orchestrator.run("Refund order ORD-1002")
    orchestrator.resume(state.run_id, True, "alice")
    events = [r.event for r in orchestrator.audit.records(state.run_id)]
    for expected in (
        "run_started",
        "plan_created",
        "risk_assessed",
        "approval_requested",
        "approval_decided",
        "tool_executed",
        "run_completed",
    ):
        assert expected in events
    assert events.index("approval_requested") < events.index("approval_decided")
    decided = orchestrator.audit.records(state.run_id, "approval_decided")[0]
    assert decided.actor == "human:alice"
    assert orchestrator.audit.verify()[0]
