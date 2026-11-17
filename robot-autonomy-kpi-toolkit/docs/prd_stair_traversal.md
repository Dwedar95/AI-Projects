# PRD (example): Stair traversal inside the ODD for the legged platform

Status: example document for this repository. It shows how a product requirement is tied to the
ODD file, the KPI definitions and the release gates of the toolkit. Targets below are example
targets, not results. Numbers marked "synthetic demo" come from `examples/run_demo.py`.

| Field | Value |
| --- | --- |
| Backlog id | AUT-101 (`examples/backlog.yaml`) |
| Owner | Product, autonomy |
| Affected platform | legged |
| ODD change | `terrain.allowed` gains `stairs`, `terrain.max_slope_deg` rises from 15 to 35 |
| KPIs moved | `mission_success_rate`, `autonomy_uptime` |

## 1. Problem

ODD 1.0 (`examples/odd_ground_legged_v1.yaml`) allows flat ground, gravel and grass with slopes up
to 15 degrees. Patrol routes on campuses and construction sites include stairs. Today an operator
takes over at every staircase, or the route is planned around it.

In the synthetic demo log of the baseline build, 22 of 80 missions are outside ODD 1.0, and the
ODD check reports 13 terrain violation spans, all of them missions that declare `stairs`. These
missions are excluded from the release KPIs, so the current KPIs say nothing about stairs.

## 2. Goal and non-goals

Goal: the legged platform traverses stairs autonomously, inside a documented ODD, with the safety
gate active, and the release gates hold on the extended ODD.

Non-goals:

- Stairs for the ground (wheeled) and aerial platforms.
- Spiral stairs, open risers, ladders, escalators.
- Any change to the safety gate's withhold criteria. The gate stays as it is and must remain
  the final authority over published commands.

## 3. Users and scenarios

- Site operator: schedules a patrol that crosses two floors and does not want to stand by for a
  takeover.
- Safety engineer: needs evidence that stair traversal stays within separation and speed limits.
- Release manager: needs a pass or fail answer per build, with the uncertainty stated.

## 4. Requirements

| Id | Requirement | Verified by |
| --- | --- | --- |
| R1 | ODD 1.1 lists `stairs` and a slope limit of 35 degrees, and passes `autonomy-kpi validate-odd`. | CLI exit code 0 |
| R2 | Missions on stairs with at least 95 % of autonomous samples inside ODD 1.1 are counted in the KPIs. | `analyse_build` mission count |
| R3 | Speed on stairs stays within `speed.max_mps`. Violations appear as `speed` spans in the ODD check. | ODD check report |
| R4 | Every motion command on stairs passes through the safety gate: each `command_proposed` has exactly one `command_published` or `command_withheld`. | `validate_mission` returns no problems |
| R5 | A fall or loss of footing is logged as a `fault` with severity `critical` and ends the mission with outcome `failed`. | Log schema |
| R6 | Each new ODD cell `environment/stairs/lighting band` is covered by at least 5 in-ODD missions before release. | `odd_coverage(..., min_missions_per_cell=5)` |

## 5. Success metrics and release gates

All metrics use the definitions in [kpi_glossary.md](kpi_glossary.md), computed on missions inside
ODD 1.1.

| KPI | Example target on stairs missions | Release gate (candidate vs last release, whole ODD) |
| --- | --- | --- |
| `mission_success_rate` | at least 0.85 | relative regression at most 5 %, absolute minimum 0.80 |
| `autonomy_uptime` | at least 0.90 | relative regression at most 3 % |
| `interventions_per_hour` | at most 2.0 | relative regression at most 15 % |
| `gate_withhold_rate` | at most 0.10 | relative regression at most 25 %, absolute maximum 0.10 |
| `min_separation_m` | at least 0.25 | relative regression at most 20 %, absolute minimum 0.25 |

The gate column matches `examples/gates.yaml`. A gate fails when the regression exceeds the
tolerance and the statistical test is significant at alpha 0.05, or when an absolute limit is
violated.

Expected side effect: adding stairs missions to the KPI population may lower fleet-level
`mission_success_rate` and `autonomy_uptime` for the first release, because harder missions
enter the denominator. The comparison baseline for that release must therefore be recomputed on
ODD 1.1, not taken from the ODD 1.0 report.

## 6. Evidence plan

1. Simulation campaign, then closed test site, then pilot customer site.
2. Sample size: the synthetic demo shows how wide the intervals are with about 60 in-ODD missions
   per build (for example a 95 % interval of 0.55 to 1.85 interventions per hour around 1.13).
   A difference of 30 % in intervention rate is not detectable at that size. Plan for at least
   150 stairs missions per build, and re-check the interval width before deciding.
3. Report: `autonomy-kpi report` output attached to the release ticket.

## 7. Risks

| Risk | Mitigation |
| --- | --- |
| Falls damage hardware or injure bystanders | Closed test site first, tethered runs, e-stop operator present, gate limits unchanged |
| KPI dilution hides a regression on flat terrain | Report KPIs per terrain in addition to fleet level (roadmap item) |
| Gate becomes too conservative on stairs and uptime drops | Track `gate_withhold_rate` per terrain, see AUT-103 |
| Localisation degrades in stairwells | AUT-102 scheduled before this feature |

## 8. Open questions

- Is descending in scope for the first release, or ascending only?
- Which lighting bands are required in stairwells? ODD 1.0 excludes lighting below 50 lux.
- Who signs off the ODD change: product, safety, or both?
