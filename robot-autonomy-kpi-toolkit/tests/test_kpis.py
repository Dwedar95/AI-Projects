import math

import pytest

from autonomy_kpi.events import Event
from autonomy_kpi.kpis import (
    KPI_BY_NAME,
    KPI_NAMES,
    bootstrap_ci,
    compute_kpis,
    summarise_log,
    summarise_mission,
)

from .conftest import make_summary, mission_events


def test_kpi_values_on_hand_built_missions():
    missions = [
        make_summary("m1"),
        make_summary(
            "m2",
            outcome="failed",
            interventions=0,
            distance_m=1000.0,
            duration_s=1800.0,
            autonomous_s=1800.0,
            withheld_by_gate=10,
            proposed=500,
            min_separation_m=0.4,
            loc_error_sum_m=54.0,
            loc_error_samples=360,
            faults=2,
        ),
    ]
    k = {name: KPI_BY_NAME[name].compute(missions) for name in KPI_NAMES}
    assert k["mission_success_rate"] == pytest.approx(0.5)
    assert k["interventions_per_hour"] == pytest.approx(2 / 1.5)
    assert k["interventions_per_km"] == pytest.approx(2 / 3.0)
    assert k["mean_distance_between_interventions_km"] == pytest.approx(1.5)
    assert k["gate_withhold_rate"] == pytest.approx(60 / 1500)
    assert k["min_separation_m"] == pytest.approx(0.4)
    assert k["autonomy_uptime"] == pytest.approx(5040 / 5400)
    assert k["localisation_error_mean_m"] == pytest.approx(90 / 1080)
    assert k["faults_per_hour"] == pytest.approx(3 / 1.5)


def test_mdbi_is_infinite_without_interventions():
    missions = [make_summary(interventions=0), make_summary("m2", interventions=0)]
    assert math.isinf(KPI_BY_NAME["mean_distance_between_interventions_km"].compute(missions))
    assert KPI_BY_NAME["interventions_per_km"].compute(missions) == 0.0


def test_summarise_mission_from_events():
    events = mission_events(
        [
            {},
            {"mode": "manual", "separation_m": 0.7},
            {"mode": "stopped", "localisation_error_m": 0.11},
            {},
        ]
    )
    extra = [
        Event(
            mission_id="m1",
            build="b1",
            t=6,
            type="intervention",
            kind="teleop_takeover",
            duration_s=5,
            cause="obstacle",
        ),
        Event(mission_id="m1", build="b1", t=7, type="fault", code="X", severity="minor"),
    ]
    for k, (etype, reason) in enumerate(
        [
            ("command_published", None),
            ("command_withheld", "manual_control"),
            ("command_withheld", "reachability_margin"),
        ]
    ):
        cid = f"c{k}"
        extra.append(
            Event(
                mission_id="m1",
                build="b1",
                t=2 + k,
                type="command_proposed",
                command_id=cid,
                speed_mps=1.0,
            )
        )
        fields = {"reason": reason} if reason else {}
        extra.append(
            Event(mission_id="m1", build="b1", t=2 + k, type=etype, command_id=cid, **fields)
        )
    s = summarise_mission(sorted([*events, *extra], key=lambda e: e.t))
    assert (s.duration_s, s.distance_m, s.outcome) == (25.0, 20.0, "success")
    assert s.autonomous_s == pytest.approx(10.0)  # two of four 5 s samples are autonomous
    assert (s.interventions, s.faults) == (1, 1)
    assert (s.proposed, s.withheld, s.withheld_by_gate) == (3, 2, 1)
    assert s.min_separation_m == pytest.approx(0.7)
    assert s.loc_error_sum_m == pytest.approx(0.05 * 3 + 0.11) and s.loc_error_samples == 4


def test_summaries_of_generated_log_are_consistent(baseline_events):
    summaries = summarise_log(baseline_events)
    assert len(summaries) == 24
    for s in summaries:
        assert 0 <= s.autonomous_s <= s.duration_s
        assert 0 <= s.withheld_by_gate <= s.withheld <= s.proposed
        assert s.distance_m > 0 and s.min_separation_m > 0
        assert s.outcome in ("success", "aborted", "failed")


def test_bootstrap_interval_contains_estimate_and_is_reproducible(baseline_events):
    summaries = summarise_log(baseline_events)
    kpis = compute_kpis(summaries, n_resamples=300, seed=1)
    again = compute_kpis(summaries, n_resamples=300, seed=1)
    assert set(kpis) == set(KPI_NAMES)
    for name in (
        "mission_success_rate",
        "interventions_per_hour",
        "autonomy_uptime",
        "gate_withhold_rate",
        "localisation_error_mean_m",
    ):
        k = kpis[name]
        assert k.ci_low <= k.value <= k.ci_high
        assert (k.ci_low, k.ci_high) == (again[name].ci_low, again[name].ci_high)
        assert k.n_missions == 24


def test_bootstrap_interval_narrows_with_more_missions():
    def spread(n):
        missions = [make_summary(f"m{i}", interventions=i % 4) for i in range(n)]
        lo, hi = bootstrap_ci(missions, KPI_BY_NAME["interventions_per_hour"].compute, 500, seed=2)
        return hi - lo

    assert spread(200) < spread(20)


def test_bootstrap_edge_cases():
    stat = KPI_BY_NAME["mission_success_rate"].compute
    assert all(math.isnan(v) for v in bootstrap_ci([make_summary()], stat))
    identical = [make_summary(f"m{i}") for i in range(10)]
    assert bootstrap_ci(identical, stat, 200) == (1.0, 1.0)
    with pytest.raises(ValueError):
        compute_kpis([])
