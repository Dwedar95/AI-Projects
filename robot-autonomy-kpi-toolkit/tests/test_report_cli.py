import re

import pytest

from autonomy_kpi.cli import main
from autonomy_kpi.events import write_log
from autonomy_kpi.gates import load_gate_config
from autonomy_kpi.kpis import KPI_DEFINITIONS, KPI_NAMES
from autonomy_kpi.prioritisation import load_backlog
from autonomy_kpi.report import (
    analyse_build,
    analyse_release,
    fmt,
    render_dashboard_html,
    render_dashboard_png,
    render_markdown,
)

from .conftest import EXAMPLES

DOCS = EXAMPLES.parent / "docs"


@pytest.fixture(scope="module")
def analysis(baseline_events, regressed_events, odd):
    return analyse_release(
        baseline_events,
        regressed_events,
        load_gate_config(EXAMPLES / "gates.yaml"),
        odd=odd,
        backlog=load_backlog(EXAMPLES / "backlog.yaml"),
        n_resamples=200,
    )


def test_analysis_uses_only_in_odd_missions(analysis, baseline_events):
    base = analysis.baseline
    assert base.synthetic is True and base.build == "v1.4.0"
    assert 0 < len(base.missions) < base.missions_total == 24
    without_odd = analyse_build(baseline_events, n_resamples=50)
    assert len(without_odd.missions) == 24 and without_odd.coverage is None


def test_regressed_candidate_fails_release(analysis):
    assert analysis.verdict == "FAIL"
    failed = {g.kpi for g in analysis.gates if g.verdict == "FAIL"}
    assert {"gate_withhold_rate", "localisation_error_mean_m"} <= failed
    top = analysis.ranked_backlog[0]
    assert top.rank == 1 and analysis.ranked_backlog[0].score >= analysis.ranked_backlog[1].score


def test_markdown_report_content(analysis):
    md = render_markdown(analysis)
    assert md.startswith("# Release readiness: v1.5.0-rc2 vs v1.4.0")
    assert "**Verdict: FAIL**" in md and "SYNTHETIC" in md
    for name in KPI_NAMES:
        assert f"| {name} |" in md
    for section in (
        "## KPIs",
        "## Regression gates",
        "## Operational Design Domain",
        "## Backlog items linked to gated KPIs",
    ):
        assert section in md
    assert chr(0x2014) not in md and chr(0x2013) not in md


def test_dashboard_png_and_html(analysis):
    png = render_dashboard_png(analysis)
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 20_000
    page = render_dashboard_html(analysis)
    assert page.startswith("<!doctype html>") and "data:image/png;base64," in page
    assert "SYNTHETIC" in page and page.count("<tr class=") == len(analysis.gates)


def test_fmt():
    assert fmt(float("nan")) == "n/a" and fmt(float("inf")) == "inf"
    assert fmt(0.12345) == "0.123" and fmt(12345.6) == "12346"


def test_glossary_documents_every_kpi_and_nothing_else():
    text = (DOCS / "kpi_glossary.md").read_text()
    documented = set(re.findall(r"^### `([a-z_]+)`", text, flags=re.MULTILINE))
    assert documented == set(KPI_NAMES)
    for d in KPI_DEFINITIONS:
        assert f"Unit: {d.unit}" in text


def test_cli_end_to_end(tmp_path, baseline_events, regressed_events, capsys):
    base, cand = tmp_path / "base.jsonl", tmp_path / "cand.jsonl"
    write_log(baseline_events, base)
    write_log(regressed_events, cand)
    odd_path, gates_path = str(EXAMPLES / "odd_ground_legged_v1.yaml"), str(EXAMPLES / "gates.yaml")

    assert main(["validate-odd", odd_path]) == 0
    bad = tmp_path / "bad.yaml"
    bad.write_text("odd: {name: x}\n")
    assert main(["validate-odd", str(bad)]) == 1

    assert main(["kpis", str(base), "--odd", odd_path]) == 0
    out = capsys.readouterr().out
    assert "SYNTHETIC" in out and "mission_success_rate" in out

    common = [
        "--baseline",
        str(base),
        "--candidate",
        str(cand),
        "--gates",
        gates_path,
        "--odd",
        odd_path,
    ]
    assert main(["gate", *common]) == 1  # FAIL maps to a non-zero exit code for CI
    assert (
        main(
            [
                "gate",
                "--baseline",
                str(base),
                "--candidate",
                str(base),
                "--gates",
                gates_path,
                "--odd",
                odd_path,
            ]
        )
        == 0
    )
    out_dir = tmp_path / "out"
    assert (
        main(
            ["report", *common, "--backlog", str(EXAMPLES / "backlog.yaml"), "--out", str(out_dir)]
        )
        == 0
    )
    assert {p.name for p in out_dir.iterdir()} == {
        "release_readiness.md",
        "dashboard.png",
        "dashboard.html",
    }
    assert (
        main(
            [
                "prioritise",
                str(EXAMPLES / "backlog.yaml"),
                "--method",
                "wsjf",
                "--failing",
                "gate_withhold_rate",
                "--boost",
                "1.5",
            ]
        )
        == 0
    )
    assert "addresses gate_withhold_rate" in capsys.readouterr().out


def test_cli_generate(tmp_path, capsys):
    out = tmp_path / "gen.jsonl"
    assert (
        main(
            ["generate", "--profile", "better", "--missions", "2", "--seed", "3", "--out", str(out)]
        )
        == 0
    )
    assert "synthetic" in capsys.readouterr().out
    assert main(["kpis", str(out)]) == 0
