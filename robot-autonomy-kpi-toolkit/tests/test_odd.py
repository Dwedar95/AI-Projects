import copy

import pytest
import yaml

from autonomy_kpi.events import group_by_mission
from autonomy_kpi.odd import (
    check_mission,
    is_in_odd,
    lighting_band,
    load_odd,
    odd_coverage,
    validate_odd,
)

from .conftest import EXAMPLES, mission_events


@pytest.fixture()
def odd_doc():
    return yaml.safe_load((EXAMPLES / "odd_ground_legged_v1.yaml").read_text())


def test_example_odd_is_valid(odd, odd_doc):
    spec, problems = validate_odd(odd_doc)
    assert problems == [] and spec == odd
    assert odd.lighting_bands() == ["dim", "bright_indoor", "daylight"]
    assert "stairs" not in odd.terrain.allowed


def test_validate_odd_reports_all_problems(odd_doc):
    bad = copy.deepcopy(odd_doc)
    bad["odd"]["lighting"] = {"min_lux": 500, "max_lux": 100}
    bad["odd"]["terrain"]["allowed"] = ["flat", "lava"]
    spec, problems = validate_odd(bad)
    assert spec is None
    text = " ".join(problems)
    assert "min_lux must be below max_lux" in text and "lava" in text

    missing = copy.deepcopy(odd_doc)
    del missing["odd"]["speed"]
    missing["odd"]["weather"] = {"max_wind": 10}
    missing["odd"]["connectivity"]["max_latency_ms"] = -5
    _, problems = validate_odd(missing)
    assert len(problems) == 3
    assert any(p.startswith("speed") for p in problems)
    assert any(p.startswith("weather") for p in problems)
    assert validate_odd(["not", "a", "mapping"])[1] == [
        "document must be a mapping with a top-level 'odd' key"
    ]


def test_load_odd_raises_with_details(tmp_path, odd_doc):
    odd_doc["odd"]["platforms"] = ["submarine"]
    path = tmp_path / "odd.yaml"
    path.write_text(yaml.safe_dump(odd_doc))
    with pytest.raises(ValueError, match="submarine"):
        load_odd(path)


def test_lighting_bands():
    assert [lighting_band(v) for v in (0, 49.9, 50, 4999, 5000, 1e6)] == [
        "dark",
        "dark",
        "dim",
        "bright_indoor",
        "daylight",
        "daylight",
    ]


def test_mission_inside_odd_has_no_violations(odd):
    result = check_mission(odd, mission_events([{}] * 10))
    assert result.violations == [] and result.in_odd_fraction == 1.0
    assert result.cell == ("outdoor_campus", "flat", "bright_indoor")
    assert is_in_odd(result)


def test_dynamic_violations_are_merged_into_spans(odd):
    states = [{}] * 10
    states[2] = {"speed_mps": 2.4}
    states[3] = {"speed_mps": 2.0}
    states[6] = {"link_latency_ms": 900.0}
    states[8] = {"mode": "manual", "speed_mps": 5.0}  # manual samples are not checked
    result = check_mission(odd, mission_events(states))
    spans = {(v.dimension, v.t_start, v.t_end, v.worst_value) for v in result.violations}
    assert spans == {("speed", 15.0, 20.0, 2.4), ("connectivity", 35.0, 35.0, 900.0)}
    assert result.autonomous_samples == 9 and result.in_odd_samples == 6
    assert result.in_odd_fraction == pytest.approx(6 / 9)
    assert not is_in_odd(result)


def test_speed_tolerance_and_two_sided_lighting(odd):
    within = check_mission(odd, mission_events([{"speed_mps": 1.55}]))
    assert within.violations == []
    dark = check_mission(odd, mission_events([{"lighting_lux": 5.0}, {"lighting_lux": 12.0}]))
    assert [(v.dimension, v.worst_value) for v in dark.violations] == [("lighting", 5.0)]
    assert dark.in_odd_fraction == 0.0 and dark.cell[2] == "dark"


def test_declared_conditions_outside_odd(odd):
    result = check_mission(odd, mission_events([{}] * 4, terrain="stairs"))
    assert [v.dimension for v in result.violations] == ["terrain"]
    assert result.in_odd_fraction == 0.0


def test_coverage_counts_only_in_odd_missions(odd):
    results = [
        check_mission(odd, mission_events([{}] * 4, mission_id="a")),
        check_mission(odd, mission_events([{}] * 4, mission_id="b")),
        check_mission(odd, mission_events([{}] * 4, mission_id="c", terrain="stairs")),
        check_mission(
            odd,
            mission_events(
                [{"speed_mps": 3.0}] * 4,
                mission_id="d",
                environment="construction_site",
                terrain="gravel",
            ),
        ),
    ]
    cov = odd_coverage(odd, results)
    assert cov.total_cells == 27 and cov.covered_cells == 1
    assert cov.fraction == pytest.approx(1 / 27)
    assert cov.missions_per_cell[("outdoor_campus", "flat", "bright_indoor")] == 2
    assert cov.out_of_odd_missions == 1  # the stairs mission
    assert ("construction_site", "gravel", "bright_indoor") in cov.uncovered
    assert odd_coverage(odd, results, min_missions_per_cell=3).covered_cells == 0


def test_odd_check_on_generated_log(odd, baseline_events):
    groups = group_by_mission(baseline_events)
    results = [check_mission(odd, groups[m]) for m in sorted(groups)]
    stairs = [r for r in results if r.cell[1] == "stairs"]
    assert stairs and all(not is_in_odd(r) for r in stairs)
    assert any(is_in_odd(r) for r in results)
    assert all(0.0 <= r.in_odd_fraction <= 1.0 for r in results)
