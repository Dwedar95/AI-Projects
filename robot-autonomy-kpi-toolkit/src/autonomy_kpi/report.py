"""Release analysis pipeline and report generators (Markdown, PNG, static HTML)."""

from __future__ import annotations

import base64
import html
import io
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .events import Event, group_by_mission
from .gates import GateConfig, GateResult, evaluate_gates, overall_verdict
from .kpis import KPI_DEFINITIONS, KpiValue, MissionSummary, compute_kpis, summarise_mission
from .odd import MissionOddResult, OddCoverage, OddSpec, check_mission, is_in_odd, odd_coverage
from .prioritisation import BacklogItem, RankedItem, rank_backlog


@dataclass(frozen=True)
class BuildAnalysis:
    """KPIs and ODD results of one build."""

    build: str
    synthetic: bool
    missions_total: int
    missions: list[MissionSummary]  # missions used for KPIs (inside the ODD if one is given)
    kpis: dict[str, KpiValue]
    odd_results: list[MissionOddResult]
    coverage: OddCoverage | None


@dataclass(frozen=True)
class ReleaseAnalysis:
    """Everything a release-readiness report needs."""

    baseline: BuildAnalysis
    candidate: BuildAnalysis
    gates: list[GateResult]
    verdict: str
    ranked_backlog: list[RankedItem]
    odd: OddSpec | None
    alpha: float


def analyse_build(
    events: list[Event],
    odd: OddSpec | None = None,
    n_resamples: int = 1000,
    seed: int = 0,
    min_in_odd_fraction: float = 0.95,
) -> BuildAnalysis:
    """Summarise a build's log. With an ODD, KPIs use only missions that stayed inside it."""
    groups = group_by_mission(events)
    if not groups:
        raise ValueError("log contains no missions")
    summaries = {mid: summarise_mission(evts) for mid, evts in groups.items()}
    builds = sorted({s.build for s in summaries.values()})
    synthetic = all(bool(e.get("synthetic")) for evts in groups.values() for e in evts[:1])
    odd_results: list[MissionOddResult] = []
    coverage = None
    used = list(summaries.values())
    if odd is not None:
        odd_results = [check_mission(odd, groups[mid]) for mid in sorted(groups)]
        coverage = odd_coverage(odd, odd_results, min_in_odd_fraction=min_in_odd_fraction)
        inside = {r.mission_id for r in odd_results if is_in_odd(r, min_in_odd_fraction)}
        used = [summaries[mid] for mid in sorted(inside)]
        if not used:
            raise ValueError("no mission stayed inside the ODD")
    used.sort(key=lambda s: s.mission_id)
    return BuildAnalysis(
        build=" + ".join(builds),
        synthetic=synthetic,
        missions_total=len(summaries),
        missions=used,
        kpis=compute_kpis(used, n_resamples=n_resamples, seed=seed),
        odd_results=odd_results,
        coverage=coverage,
    )


def analyse_release(
    baseline_events: list[Event],
    candidate_events: list[Event],
    gate_config: GateConfig,
    odd: OddSpec | None = None,
    backlog: list[BacklogItem] | None = None,
    method: str = "rice",
    gate_boost: float = 1.0,
    n_resamples: int = 1000,
    seed: int = 0,
) -> ReleaseAnalysis:
    """Run the full comparison: KPIs, ODD checks, gates and backlog ranking."""
    base = analyse_build(baseline_events, odd, n_resamples, seed)
    cand = analyse_build(candidate_events, odd, n_resamples, seed)
    gates = evaluate_gates(base.missions, cand.missions, gate_config)
    failing = tuple(g.kpi for g in gates if g.verdict != "PASS")
    ranked = rank_backlog(backlog, method, failing, gate_boost) if backlog else []  # type: ignore[arg-type]
    return ReleaseAnalysis(
        base, cand, gates, overall_verdict(gates), ranked, odd, gate_config.alpha
    )


def fmt(value: float, digits: int = 3) -> str:
    """Format a KPI number: compact, with ``inf`` and ``n/a`` handled."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.{digits}g}" if abs(value) < 1000 else f"{value:.0f}"


def _kpi_cell(k: KpiValue) -> str:
    return f"{fmt(k.value)} [{fmt(k.ci_low)}, {fmt(k.ci_high)}]"


def _odd_section(label: str, build: BuildAnalysis) -> list[str]:
    cov = build.coverage
    if cov is None:
        return []
    by_dim = Counter(v.dimension for r in build.odd_results for v in r.violations)
    lines = [
        f"### {label}: {build.build}",
        "",
        f"- Missions inside the ODD (used for KPIs): {len(build.missions)} of "
        f"{build.missions_total}",
        f"- ODD cells covered: {cov.covered_cells} of {cov.total_cells} "
        f"({100 * cov.fraction:.0f} %)",
        f"- Missions in conditions outside the ODD cells: {cov.out_of_odd_missions}",
    ]
    if by_dim:
        spans = ", ".join(f"{dim}: {n}" for dim, n in sorted(by_dim.items()))
        lines.append(f"- Violation spans by dimension: {spans}")
    if cov.uncovered:
        shown = ", ".join("/".join(c) for c in cov.uncovered[:8])
        more = f" (and {len(cov.uncovered) - 8} more)" if len(cov.uncovered) > 8 else ""
        lines.append(f"- Uncovered cells (environment/terrain/lighting): {shown}{more}")
    return [*lines, ""]


def render_markdown(analysis: ReleaseAnalysis, top_n: int = 5) -> str:
    """Release-readiness report as Markdown."""
    base, cand = analysis.baseline, analysis.candidate
    lines = [
        f"# Release readiness: {cand.build} vs {base.build}",
        "",
        f"**Verdict: {analysis.verdict}**",
        "",
    ]
    if base.synthetic and cand.synthetic:
        lines += [
            "> Data source: SYNTHETIC mission logs from the seeded generator. Not field data.",
            "",
        ]
    counts = Counter(g.verdict for g in analysis.gates)
    lines += [
        f"Gates: {counts.get('PASS', 0)} pass, {counts.get('WARN', 0)} warn, "
        f"{counts.get('FAIL', 0)} fail (alpha = {analysis.alpha}).",
        "",
        "## KPIs",
        "",
        f"Point estimate with 95 % bootstrap interval over missions "
        f"(baseline n = {len(base.missions)}, candidate n = {len(cand.missions)}).",
        "",
        "| KPI | Unit | Better | Baseline | Candidate |",
        "| --- | --- | --- | --- | --- |",
    ]
    for d in KPI_DEFINITIONS:
        direction = "higher" if d.higher_is_better else "lower"
        lines.append(
            f"| {d.name} | {d.unit} | {direction} | {_kpi_cell(base.kpis[d.name])} | "
            f"{_kpi_cell(cand.kpis[d.name])} |"
        )
    lines += [
        "",
        "## Regression gates",
        "",
        "| KPI | Baseline | Candidate | Regression | Test | p (worse) | Verdict | Reason |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for g in analysis.gates:
        regression = (
            "n/a"
            if math.isnan(g.relative_regression)
            else (
                fmt(g.relative_regression)
                if math.isinf(g.relative_regression)
                else f"{100 * g.relative_regression:+.1f} %"
            )
        )
        lines.append(
            f"| {g.kpi} | {fmt(g.baseline)} | {fmt(g.candidate)} | {regression} | {g.test} | "
            f"{g.p_value:.4f} | {g.verdict} | {g.reason} |"
        )
    lines += [
        "",
        "Regression is the change in the unfavourable direction relative to the "
        "baseline. Positive means worse.",
        "",
    ]
    if analysis.odd is not None:
        lines += [
            "## Operational Design Domain",
            "",
            f"ODD: {analysis.odd.name} (version {analysis.odd.version}).",
            "",
            *_odd_section("Baseline", base),
            *_odd_section("Candidate", cand),
        ]
    if analysis.ranked_backlog:
        failing = [g.kpi for g in analysis.gates if g.verdict != "PASS"]
        lines += ["## Backlog items linked to gated KPIs", ""]
        if failing:
            lines += [f"KPIs with a WARN or FAIL gate: {', '.join(failing)}.", ""]
        lines += ["| Rank | Id | Title | Score | Addresses |", "| --- | --- | --- | --- | --- |"]
        for r in analysis.ranked_backlog[:top_n]:
            addresses = ", ".join(r.addresses) or "none of the above"
            lines.append(
                f"| {r.rank} | {r.item.id} | {r.item.title} | {fmt(r.score, 4)} | {addresses} |"
            )
        lines.append("")
    return "\n".join(lines)


def render_dashboard_png(analysis: ReleaseAnalysis) -> bytes:
    """KPI dashboard as PNG: one panel per KPI, baseline against candidate with intervals."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base, cand = analysis.baseline, analysis.candidate
    verdicts = {g.kpi: g.verdict for g in analysis.gates}
    colours = {"baseline": "#6b7280", "candidate": "#2563eb"}
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for ax, d in zip(axes.ravel(), KPI_DEFINITIONS, strict=True):
        for x, (label, build) in enumerate((("baseline", base), ("candidate", cand))):
            k = build.kpis[d.name]
            if not math.isfinite(k.value):
                ax.text(x, 0, "inf", ha="center", va="bottom")
                continue
            ax.bar(x, k.value, color=colours[label], width=0.6)
            if math.isfinite(k.ci_low) and math.isfinite(k.ci_high):
                ax.errorbar(
                    x,
                    k.value,
                    yerr=[[max(k.value - k.ci_low, 0)], [max(k.ci_high - k.value, 0)]],
                    color="black",
                    capsize=4,
                    linewidth=1,
                )
        ax.set_xticks([0, 1], [base.build, cand.build], fontsize=8)
        gate = verdicts.get(d.name)
        suffix = f"  [{gate}]" if gate else ""
        ax.set_title(
            f"{d.name}{suffix}", fontsize=9, color="#b91c1c" if gate == "FAIL" else "black"
        )
        better = "higher" if d.higher_is_better else "lower"
        ax.set_ylabel(f"{d.unit} ({better} is better)", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    note = " (SYNTHETIC data)" if base.synthetic and cand.synthetic else ""
    fig.suptitle(
        f"Autonomy KPIs: {cand.build} vs {base.build}, verdict {analysis.verdict}{note}",
        fontsize=12,
    )
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110)
    plt.close(fig)
    return buffer.getvalue()


def render_dashboard_html(analysis: ReleaseAnalysis) -> str:
    """Self-contained static HTML dashboard (tables plus the PNG embedded as base64)."""
    png = base64.b64encode(render_dashboard_png(analysis)).decode("ascii")
    base, cand = analysis.baseline, analysis.candidate
    esc = html.escape
    kpi_rows = "".join(
        f"<tr><td>{esc(d.name)}</td><td>{esc(d.unit)}</td>"
        f"<td>{esc(_kpi_cell(base.kpis[d.name]))}</td>"
        f"<td>{esc(_kpi_cell(cand.kpis[d.name]))}</td></tr>"
        for d in KPI_DEFINITIONS
    )
    gate_rows = "".join(
        f"<tr class='{g.verdict.lower()}'><td>{esc(g.kpi)}</td><td>{esc(g.test)}</td>"
        f"<td>{g.p_value:.4f}</td><td>{g.verdict}</td><td>{esc(g.reason)}</td></tr>"
        for g in analysis.gates
    )
    note = (
        "<p class='note'>Data source: SYNTHETIC mission logs. Not field data.</p>"
        if base.synthetic and cand.synthetic
        else ""
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Autonomy KPI dashboard: {esc(cand.build)} vs {esc(base.build)}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #111827; }}
table {{ border-collapse: collapse; margin-bottom: 1.5rem; }}
th, td {{ border: 1px solid #d1d5db; padding: 4px 10px; font-size: 14px; text-align: left; }}
th {{ background: #f3f4f6; }}
tr.fail td {{ background: #fee2e2; }} tr.warn td {{ background: #fef3c7; }}
.note {{ color: #92400e; font-weight: 600; }}
img {{ max-width: 100%; }}
</style></head><body>
<h1>Autonomy KPI dashboard</h1>
<p>Candidate <b>{esc(cand.build)}</b> against baseline <b>{esc(base.build)}</b>.
Verdict: <b>{analysis.verdict}</b>.</p>
{note}
<img alt="KPI panels, baseline against candidate" src="data:image/png;base64,{png}">
<h2>KPIs (95 % bootstrap interval)</h2>
<table><tr><th>KPI</th><th>Unit</th><th>Baseline</th><th>Candidate</th></tr>{kpi_rows}</table>
<h2>Regression gates</h2>
<table><tr><th>KPI</th><th>Test</th><th>p (worse)</th><th>Verdict</th><th>Reason</th></tr>
{gate_rows}</table>
</body></html>
"""


def write_reports(analysis: ReleaseAnalysis, out_dir: str | Path) -> dict[str, Path]:
    """Write ``release_readiness.md``, ``dashboard.png`` and ``dashboard.html``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "markdown": out / "release_readiness.md",
        "png": out / "dashboard.png",
        "html": out / "dashboard.html",
    }
    paths["markdown"].write_text(render_markdown(analysis), encoding="utf-8")
    paths["png"].write_bytes(render_dashboard_png(analysis))
    paths["html"].write_text(render_dashboard_html(analysis), encoding="utf-8")
    return paths
