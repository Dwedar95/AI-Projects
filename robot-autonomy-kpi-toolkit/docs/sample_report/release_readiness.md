# Release readiness: v1.5.0-rc2 vs v1.4.0

**Verdict: FAIL**

> Data source: SYNTHETIC mission logs from the seeded generator. Not field data.

Gates: 1 pass, 1 warn, 5 fail (alpha = 0.05).

## KPIs

Point estimate with 95 % bootstrap interval over missions (baseline n = 58, candidate n = 63).

| KPI | Unit | Better | Baseline | Candidate |
| --- | --- | --- | --- | --- |
| mission_success_rate | fraction | higher | 0.879 [0.793, 0.948] | 0.825 [0.714, 0.905] |
| interventions_per_hour | 1/h | lower | 1.13 [0.546, 1.85] | 3.63 [2.82, 4.56] |
| interventions_per_km | 1/km | lower | 0.322 [0.156, 0.528] | 1.1 [0.845, 1.38] |
| mean_distance_between_interventions_km | km | higher | 3.11 [1.9, 6.39] | 0.913 [0.725, 1.18] |
| gate_withhold_rate | fraction | lower | 0.0213 [0.0173, 0.025] | 0.0587 [0.0546, 0.063] |
| min_separation_m | m | higher | 0.359 [0.359, 0.383] | 0.354 [0.354, 0.389] |
| autonomy_uptime | fraction | higher | 0.969 [0.963, 0.974] | 0.91 [0.897, 0.921] |
| localisation_error_mean_m | m | lower | 0.0779 [0.0725, 0.084] | 0.127 [0.116, 0.138] |
| faults_per_hour | 1/h | lower | 1.53 [0.909, 2.28] | 1.21 [0.68, 1.89] |

## Regression gates

| KPI | Baseline | Candidate | Regression | Test | p (worse) | Verdict | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mission_success_rate | 0.879 | 0.825 | +6.1 % | fisher | 0.2831 | WARN | regression beyond tolerance, not significant (more data) |
| interventions_per_hour | 1.13 | 3.63 | +222.7 % | poisson_rate | 0.0000 | FAIL | regression beyond tolerance, statistically significant |
| mean_distance_between_interventions_km | 3.11 | 0.913 | +70.6 % | poisson_rate | 0.0000 | FAIL | regression beyond tolerance, statistically significant |
| gate_withhold_rate | 0.0213 | 0.0587 | +176.1 % | mann_whitney | 0.0000 | FAIL | regression beyond tolerance, statistically significant |
| min_separation_m | 0.359 | 0.354 | +1.4 % | mann_whitney | 0.3778 | PASS | no regression beyond tolerance |
| autonomy_uptime | 0.969 | 0.91 | +6.1 % | mann_whitney | 0.0000 | FAIL | regression beyond tolerance, statistically significant |
| localisation_error_mean_m | 0.0779 | 0.127 | +62.5 % | mann_whitney | 0.0000 | FAIL | regression beyond tolerance, statistically significant |

Regression is the change in the unfavourable direction relative to the baseline. Positive means worse.

## Operational Design Domain

ODD: campus-and-site-patrol (version 1.0).

### Baseline: v1.4.0

- Missions inside the ODD (used for KPIs): 58 of 80
- ODD cells covered: 16 of 27 (59 %)
- Missions in conditions outside the ODD cells: 22
- Violation spans by dimension: connectivity: 1, lighting: 9, slope: 30, terrain: 13
- Uncovered cells (environment/terrain/lighting): indoor_warehouse/flat/bright_indoor, indoor_warehouse/flat/daylight, indoor_warehouse/gravel/dim, indoor_warehouse/gravel/bright_indoor, indoor_warehouse/gravel/daylight, indoor_warehouse/grass/dim, indoor_warehouse/grass/bright_indoor, indoor_warehouse/grass/daylight (and 3 more)

### Candidate: v1.5.0-rc2

- Missions inside the ODD (used for KPIs): 63 of 80
- ODD cells covered: 15 of 27 (56 %)
- Missions in conditions outside the ODD cells: 17
- Violation spans by dimension: connectivity: 1, lighting: 6, slope: 26, terrain: 12
- Uncovered cells (environment/terrain/lighting): indoor_warehouse/flat/bright_indoor, indoor_warehouse/flat/daylight, indoor_warehouse/gravel/dim, indoor_warehouse/gravel/bright_indoor, indoor_warehouse/gravel/daylight, indoor_warehouse/grass/dim, indoor_warehouse/grass/bright_indoor, indoor_warehouse/grass/daylight (and 4 more)

## Backlog items linked to gated KPIs

KPIs with a WARN or FAIL gate: mission_success_rate, interventions_per_hour, mean_distance_between_interventions_km, gate_withhold_rate, autonomy_uptime, localisation_error_mean_m.

| Rank | Id | Title | Score | Addresses |
| --- | --- | --- | --- | --- |
| 1 | AUT-102 | Relocalisation after tracking loss | 160 | localisation_error_mean_m, interventions_per_hour, mean_distance_between_interventions_km |
| 2 | AUT-106 | Sensor watchdog and graceful degradation | 160 | mission_success_rate |
| 3 | AUT-103 | Less conservative reachability margin at low speed | 140 | gate_withhold_rate, autonomy_uptime |
| 4 | AUT-107 | Operator console intervention shortcuts | 90 | autonomy_uptime |
| 5 | AUT-104 | Planner recovery behaviours for blocked paths | 87.5 | interventions_per_hour, mission_success_rate |
