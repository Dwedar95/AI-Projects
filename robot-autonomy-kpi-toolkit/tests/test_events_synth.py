import json

import pytest
from pydantic import ValidationError

from autonomy_kpi.events import (
    Event,
    group_by_mission,
    read_log,
    validate_mission,
    write_log,
)
from autonomy_kpi.synth import BASELINE, generate_mission

from .conftest import mission_events


def test_event_requires_type_specific_fields():
    with pytest.raises(ValidationError):
        Event(mission_id="m", build="b", t=1, type="command_withheld", command_id="c1")
    with pytest.raises(ValidationError):
        Event(
            mission_id="m",
            build="b",
            t=1,
            type="mission_end",
            outcome="exploded",
            distance_m=1,
            duration_s=1,
        )
    with pytest.raises(ValidationError):
        Event(mission_id="m", build="b", t=-1, type="fault", code="X", severity="minor")
    with pytest.raises(ValidationError):
        Event(mission_id="m", build="b", t=1, type="teleport")
    ok = Event(
        mission_id="m",
        build="b",
        t=1,
        type="command_withheld",
        command_id="c1",
        reason="unsafe_separation",
    )
    assert ok.get("reason") == "unsafe_separation" and ok.get("missing", 7) == 7


def test_jsonl_roundtrip_and_bad_line(tmp_path):
    events = generate_mission(BASELINE, 0, seed=1)
    path = tmp_path / "log.jsonl"
    assert write_log(events, path) == len(events)
    loaded = read_log(path)
    assert [e.model_dump() for e in loaded] == [e.model_dump() for e in events]
    first = json.loads(path.read_text().splitlines()[0])
    assert first["type"] == "mission_start" and first["synthetic"] is True
    bad = tmp_path / "bad.jsonl"
    bad.write_text(path.read_text().splitlines()[0] + '\n{"type": "state"}\n')
    with pytest.raises(ValueError, match="bad.jsonl:2"):
        read_log(bad)


def test_generator_is_deterministic_and_labelled_synthetic():
    a = generate_mission(BASELINE, 3, seed=9)
    b = generate_mission(BASELINE, 3, seed=9)
    c = generate_mission(BASELINE, 3, seed=10)
    assert [e.model_dump() for e in a] == [e.model_dump() for e in b]
    assert [e.model_dump() for e in a] != [e.model_dump() for e in c]
    assert a[0].type == "mission_start" and a[0].get("synthetic") is True


def test_generated_missions_are_structurally_valid(baseline_events, regressed_events):
    for events in (baseline_events, regressed_events):
        groups = group_by_mission(events)
        assert len(groups) == 24
        for mission in groups.values():
            assert validate_mission(mission) == []
            times = [e.t for e in mission]
            assert times == sorted(times)


def test_gate_semantics_in_generated_log(baseline_events):
    # Whenever the gate publishes a command, separation at that time is above the gate margin:
    # the state sample at the same second (if any) shows a safe separation.
    by_mission = group_by_mission(baseline_events)
    reasons = set()
    for mission in by_mission.values():
        unsafe_t = {e.t for e in mission if e.get("reason") == "unsafe_separation"}
        published_t = {e.t for e in mission if e.type == "command_published"}
        assert not unsafe_t & published_t
        reasons |= {e.get("reason") for e in mission if e.type == "command_withheld"}
    assert "unsafe_separation" in reasons and "reachability_margin" in reasons


def test_validate_mission_detects_structural_problems():
    events = mission_events([{}, {}])
    assert validate_mission(events) == []
    assert validate_mission([]) == ["mission has no events"]
    assert any("mission_end" in p for p in validate_mission(events[:-1]))
    dangling = [
        *events[:-1],
        Event(
            mission_id="m1",
            build="b1",
            t=11,
            type="command_proposed",
            command_id="c1",
            speed_mps=1.0,
        ),
        Event(mission_id="m1", build="b1", t=11, type="command_published", command_id="c9"),
        events[-1],
    ]
    problems = validate_mission(dangling)
    assert any("unknown command c9" in p for p in problems)
    assert any("no gate decision" in p for p in problems)
    backwards = mission_events([{"distance_m": 10.0}, {"distance_m": 4.0}])
    assert any("distance decreases" in p for p in validate_mission(backwards))
