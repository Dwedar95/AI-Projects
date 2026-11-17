import json

import pytest

from agentic_orchestrator.audit import GENESIS_HASH, AuditLog
from agentic_orchestrator.memory import ConversationMemory, InMemoryStateStore, JsonFileStateStore
from agentic_orchestrator.models import RunState


def _filled(path=None) -> AuditLog:
    log = AuditLog(path)
    log.append("run_a", "run_started", "orchestrator", {"goal": "x"})
    log.append("run_a", "tool_executed", "orchestrator", {"tool": "calculator"})
    log.append("run_b", "run_started", "orchestrator", {"goal": "y"})
    return log


def test_chain_links_and_verifies():
    log = _filled()
    records = log.records()
    assert records[0].prev_hash == GENESIS_HASH
    assert records[1].prev_hash == records[0].hash
    assert log.verify() == (True, "3 records verified")


def test_filters_by_run_and_event():
    log = _filled()
    assert len(log.records(run_id="run_a")) == 2
    assert [r.run_id for r in log.records(event="run_started")] == ["run_a", "run_b"]


def test_tampering_with_payload_is_detected():
    log = _filled()
    log.records()[1].payload["tool"] = "delete_customer_record"
    ok, detail = log.verify()
    assert not ok and "record 1" in detail and "hash mismatch" in detail


def test_removing_a_line_from_the_file_is_detected(tmp_path):
    path = tmp_path / "audit.jsonl"
    _filled(path)
    lines = path.read_text().splitlines()
    path.write_text("\n".join([lines[0], lines[2]]) + "\n")
    ok, detail = AuditLog(path).verify()
    assert not ok and "record 1" in detail


def test_file_log_reloads_and_continues_the_chain(tmp_path):
    path = tmp_path / "audit.jsonl"
    first = _filled(path)
    reopened = AuditLog(path)
    record = reopened.append("run_b", "run_completed", "orchestrator", {})
    assert record.seq == 3 and record.prev_hash == first.records()[-1].hash
    assert len(path.read_text().splitlines()) == 4
    assert json.loads(path.read_text().splitlines()[3])["event"] == "run_completed"
    assert AuditLog(path).verify()[0]


def test_memory_keeps_recent_turns_and_summarises_old_ones():
    memory = ConversationMemory(max_tokens=40)
    memory.add("s", "user", "First question about shipping. With extra detail " * 2)
    memory.add("s", "assistant", "Shipping takes 3 to 5 days. " * 3)
    memory.add("s", "user", "Second question about returns and refunds " * 2)
    turns = memory.turns("s")
    assert len(turns) < 3 and turns[-1][1].startswith("Second question")
    context = memory.context("s")
    assert context.startswith("Earlier (summarised): user: First question about shipping.")


def test_memory_sessions_are_isolated():
    memory = ConversationMemory()
    memory.add("a", "user", "hello")
    assert memory.context("b") == "" and memory.context(None) == ""
    with pytest.raises(ValueError):
        ConversationMemory(max_tokens=5)


def test_in_memory_store_returns_copies():
    store = InMemoryStateStore()
    state = RunState(goal="g")
    store.save(state)
    state.goal = "mutated"
    assert store.get(state.run_id).goal == "g"
    assert store.get("missing") is None


def test_json_file_store_round_trip_and_id_validation(tmp_path):
    store = JsonFileStateStore(tmp_path)
    state = RunState(goal="persist me", tainted=True)
    store.save(state)
    loaded = JsonFileStateStore(tmp_path).get(state.run_id)
    assert loaded == state
    assert [s.run_id for s in store.list()] == [state.run_id]
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(ValueError):
        store.get("../etc/passwd")
