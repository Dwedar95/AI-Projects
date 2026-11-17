"""End-to-end demo on SYNTHETIC mission logs. Prints numbers computed in this run.

Run with: python examples/run_demo.py
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from autonomy_kpi.events import group_by_mission, read_log, validate_mission, write_log
from autonomy_kpi.gates import load_gate_config
from autonomy_kpi.kpis import KPI_DEFINITIONS
from autonomy_kpi.odd import load_odd
from autonomy_kpi.prioritisation import load_backlog, rank_backlog
from autonomy_kpi.report import analyse_release, fmt, write_reports
from autonomy_kpi.synth import BASELINE, CANDIDATE_BETTER, CANDIDATE_REGRESSED, generate_log

HERE = Path(__file__).parent
OUT = HERE / "out"
SEED = 11
N_MISSIONS = 80


def main() -> None:
    OUT.mkdir(exist_ok=True)
    print("NOTE: all mission logs in this demo are SYNTHETIC (seeded generator, not field data).")

    print("\n== 1. Synthetic mission logs ==")
    logs = {}
    for profile in (BASELINE, CANDIDATE_BETTER, CANDIDATE_REGRESSED):
        path = OUT / f"{profile.name}.jsonl"
        n = write_log(generate_log(profile, N_MISSIONS, SEED), path)
        logs[profile.name] = read_log(path)
        malformed = sum(
            bool(validate_mission(e)) for e in group_by_mission(logs[profile.name]).values()
        )
        print(f"{profile.name:<11}: {N_MISSIONS} missions, {n} events, malformed: {malformed}")

    odd = load_odd(HERE / "odd_ground_legged_v1.yaml")
    gates = load_gate_config(HERE / "gates.yaml")
    backlog = load_backlog(HERE / "backlog.yaml")
    print(f"\n== 2. ODD '{odd.name}' v{odd.version} ==")
    print(f"lighting bands in ODD: {odd.lighting_bands()}")

    for cand_name in (CANDIDATE_BETTER.name, CANDIDATE_REGRESSED.name):
        analysis = analyse_release(
            logs[BASELINE.name], logs[cand_name], gates, odd=odd, backlog=backlog, seed=SEED
        )
        base, cand = analysis.baseline, analysis.candidate
        print(f"\n== 3. Release check: {cand_name} vs {BASELINE.name} ==")
        for build in (base, cand):
            cov = build.coverage
            dims = Counter(v.dimension for r in build.odd_results for v in r.violations)
            print(
                f"{build.build:<11}: {len(build.missions)}/{build.missions_total} missions inside "
                f"ODD, cells covered {cov.covered_cells}/{cov.total_cells}, "
                f"violation spans {dict(sorted(dims.items()))}"
            )
        print(f"{'KPI':<40} {'baseline [95% CI]':<28} {'candidate [95% CI]':<28}")
        for d in KPI_DEFINITIONS:
            b, c = base.kpis[d.name], cand.kpis[d.name]
            print(
                f"{d.name:<40} "
                f"{fmt(b.value) + ' [' + fmt(b.ci_low) + ', ' + fmt(b.ci_high) + ']':<28} "
                f"{fmt(c.value) + ' [' + fmt(c.ci_low) + ', ' + fmt(c.ci_high) + ']':<28}"
            )
        print("gates:")
        for g in analysis.gates:
            print(
                f"  {g.verdict:<4} {g.kpi:<40} regression {100 * g.relative_regression:+7.1f} % "
                f"p={g.p_value:.4f} ({g.test})"
            )
        print(f"overall verdict: {analysis.verdict}")
        paths = write_reports(analysis, OUT / cand_name)
        print("reports: " + ", ".join(str(p.relative_to(HERE.parent)) for p in paths.values()))

    failing = [g.kpi for g in analysis.gates if g.verdict != "PASS"]
    print(f"\n== 4. Backlog prioritisation (gates not passing on {CANDIDATE_REGRESSED.name}) ==")
    print(f"KPIs with WARN or FAIL: {failing}")
    for method in ("rice", "wsjf"):
        ranked = rank_backlog(backlog, method, failing)
        print(
            f"{method.upper()} order: " + ", ".join(f"{r.item.id} ({r.score:.1f})" for r in ranked)
        )
    boosted = rank_backlog(backlog, "wsjf", failing, gate_boost=1.5)
    print("WSJF with gate boost 1.5: " + ", ".join(f"{r.item.id} ({r.score:.1f})" for r in boosted))


if __name__ == "__main__":
    main()
