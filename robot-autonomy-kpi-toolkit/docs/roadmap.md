# Roadmap (example)

This roadmap belongs to the example product used in this repository. Items refer to
`examples/backlog.yaml`. The order follows the WSJF ranking printed by `examples/run_demo.py`,
adjusted for one dependency (AUT-102 before AUT-101). Dates are relative on purpose: the document
shows the structure, not a commitment.

## Principles

1. Every item names the KPIs it should move (see [kpi_glossary.md](kpi_glossary.md)). An item
   without a measurable KPI does not enter the roadmap.
2. Releases ship only when the regression gates pass on missions inside the ODD.
3. ODD extensions are product decisions with their own evidence plan, not side effects of
   engineering work.
4. Scores (RICE, WSJF) order the discussion. Dependencies and safety arguments can override them,
   and each override is written down.

## Now (current quarter): stabilise the release line

| Item | KPIs | Why now |
| --- | --- | --- |
| AUT-106 Sensor watchdog and graceful degradation | `faults_per_hour`, `mission_success_rate` | Highest WSJF score (7.7): small job, high risk reduction. |
| AUT-102 Relocalisation after tracking loss | `localisation_error_mean_m`, `interventions_per_hour`, `mean_distance_between_interventions_km` | Second WSJF score (5.8), top RICE score (160, tied with AUT-106). Prerequisite for stairwells. |
| AUT-103 Less conservative reachability margin at low speed | `gate_withhold_rate`, `autonomy_uptime` | Third in both rankings. Needs safety review because it touches the gate margin. |

Exit criterion: candidate build passes all gates in `examples/gates.yaml` against the last release.

## Next (following quarter): fewer interventions

| Item | KPIs | Note |
| --- | --- | --- |
| AUT-104 Planner recovery behaviours for blocked paths | `interventions_per_hour`, `interventions_per_km`, `mission_success_rate` | Confidence is only 0.5. Run a two-week spike to raise or lower it before committing. |
| AUT-107 Operator console intervention shortcuts | `autonomy_uptime` | Cheap. Shortens interventions, does not prevent them. |

## Later: extend the ODD

| Item | KPIs | Note |
| --- | --- | --- |
| AUT-101 Stair traversal inside the ODD (legged) | `mission_success_rate`, `autonomy_uptime` | See [prd_stair_traversal.md](prd_stair_traversal.md). Scheduled after AUT-102 although its score is independent of it: this is a dependency override. |
| AUT-105 Low-light perception mode | `mission_success_rate`, `min_separation_m` | Lowest score in both rankings (confidence 0.4, effort 6). Revisit when a customer requires operation below 50 lux. |

## Toolkit roadmap

Improvements to this toolkit itself, in rough priority order:

1. KPI breakdown per terrain, environment and platform, to avoid fleet-level dilution.
2. Correction for multiple comparisons across gates (for example Holm).
3. Sequential testing, so a release decision can be taken as soon as evidence suffices.
4. ODD cells with an explicit exclusion list for combinations that cannot occur.
5. Importer for rosbag2 or MCAP logs into the JSONL event format.
