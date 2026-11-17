from pathlib import Path

import pytest

from autonomy_kpi.events import Event
from autonomy_kpi.kpis import MissionSummary
from autonomy_kpi.odd import load_odd
from autonomy_kpi.synth import BASELINE, CANDIDATE_REGRESSED, generate_log

EXAMPLES = Path(__file__).parent.parent / "examples"


@pytest.fixture(scope="session")
def baseline_events() -> list[Event]:
    return generate_log(BASELINE, 24, seed=5)


@pytest.fixture(scope="session")
def regressed_events() -> list[Event]:
    return generate_log(CANDIDATE_REGRESSED, 24, seed=5)


@pytest.fixture(scope="session")
def odd():
    return load_odd(EXAMPLES / "odd_ground_legged_v1.yaml")


def make_summary(mission_id: str = "m1", **overrides) -> MissionSummary:
    """Hand-built mission summary with round numbers for exact KPI checks."""
    fields = {
        "mission_id": mission_id,
        "build": "b1",
        "platform": "legged",
        "environment": "outdoor_campus",
        "terrain": "flat",
        "outcome": "success",
        "duration_s": 3600.0,
        "distance_m": 2000.0,
        "autonomous_s": 3240.0,
        "interventions": 2,
        "proposed": 1000,
        "withheld": 60,
        "withheld_by_gate": 50,
        "min_separation_m": 0.9,
        "loc_error_sum_m": 36.0,
        "loc_error_samples": 720,
        "faults": 1,
    }
    fields.update(overrides)
    return MissionSummary(**fields)


def mission_events(states: list[dict], mission_id: str = "m1", **start_overrides) -> list[Event]:
    """Minimal hand-written mission: start, the given state samples, end."""
    start = {
        "platform": "legged",
        "environment": "outdoor_campus",
        "terrain": "flat",
        "planned_distance_m": 100.0,
        "synthetic": True,
    }
    start.update(start_overrides)
    base_state = {
        "mode": "autonomous",
        "distance_m": 0.0,
        "speed_mps": 1.0,
        "localisation_error_m": 0.05,
        "separation_m": 3.0,
        "battery_soc": 0.9,
        "lighting_lux": 1000.0,
        "link_latency_ms": 50.0,
        "slope_deg": 2.0,
    }
    events = [Event(mission_id=mission_id, build="b1", t=0, type="mission_start", **start)]
    t = 0.0
    for k, overrides in enumerate(states, start=1):
        t = 5.0 * k
        fields = {**base_state, "distance_m": 5.0 * k, **overrides}
        events.append(Event(mission_id=mission_id, build="b1", t=t, type="state", **fields))
    events.append(
        Event(
            mission_id=mission_id,
            build="b1",
            t=t + 5,
            type="mission_end",
            outcome="success",
            distance_m=5.0 * len(states),
            duration_s=t + 5,
        )
    )
    return events
