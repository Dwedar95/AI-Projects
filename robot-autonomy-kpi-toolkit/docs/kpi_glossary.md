# KPI glossary

This glossary describes exactly what `src/autonomy_kpi/kpis.py` computes. A test
(`tests/test_report_cli.py::test_glossary_documents_every_kpi_and_nothing_else`) fails if a KPI is
added to the code without an entry here, or the other way round.

## Conventions

- **Mission**: all events that share a `mission_id`, from `mission_start` to `mission_end`.
- **Fleet value**: every KPI is a ratio of sums over missions, not a mean of per-mission ratios.
  Long missions therefore weigh more than short ones.
- **Missions used**: when an ODD is supplied, only missions with at least 95 % of their autonomous
  state samples inside the ODD are used. Without an ODD all missions are used.
- **Confidence interval**: 95 % percentile bootstrap that resamples whole missions with
  replacement (1000 resamples by default, seeded). Events inside one mission are correlated, so
  resampling single events would give intervals that are too narrow.
- **Regression**: change in the unfavourable direction divided by the baseline value. Positive
  means the candidate is worse.

## KPIs

### `mission_success_rate`

Unit: fraction. Higher is better.
Missions whose `mission_end.outcome` is `success`, divided by all missions. `aborted` and `failed`
both count as not successful.
Gate test: Fisher's exact test on the 2x2 table of outcomes (one-sided, candidate worse).

### `interventions_per_hour`

Unit: 1/h. Lower is better.
Number of `intervention` events divided by total mission duration in hours (`mission_end.duration_s`
summed over missions). All intervention kinds count equally: `teleop_takeover`, `remote_assist`,
`e_stop`. The denominator is total mission time, not autonomous time only.
Gate test: exact conditional test for two Poisson rates with mission hours as exposure.

### `interventions_per_km`

Unit: 1/km. Lower is better.
Number of `intervention` events divided by total distance in km (`mission_end.distance_m` summed
over missions). Distance includes distance covered under manual control.
Gate test: exact conditional test for two Poisson rates with km as exposure.

### `mean_distance_between_interventions_km`

Unit: km. Higher is better.
Total distance in km divided by the number of `intervention` events. It is the reciprocal of
`interventions_per_km` and is reported as infinite when there is no intervention. A bootstrap
upper bound of `inf` means that some resamples contained no intervention.
Gate test: same as `interventions_per_km` (the two KPIs carry the same evidence).

### `gate_withhold_rate`

Unit: fraction. Lower is better.
`command_withheld` events whose reason is not `manual_control`, divided by `command_proposed`
events. Commands withheld because a human had control are excluded from the numerator, since they
say nothing about how conservative the safety gate is. They stay in the denominator. Lower is
better only as long as `min_separation_m` holds: a gate that never withholds is not a goal.
Gate test: Mann-Whitney U test on per-mission withhold rates.

### `min_separation_m`

Unit: m. Higher is better.
Smallest `separation_m` in any `state` sample of any mission. This is a worst-case statistic.
Its bootstrap interval is not reliable (the minimum of a resample can never be below the observed
minimum), so use the absolute limit in the gate configuration as the primary check.
Gate test: Mann-Whitney U test on per-mission minimum separations.

### `autonomy_uptime`

Unit: fraction. Higher is better.
Time in mode `autonomous` divided by total mission duration. Each `state` sample accounts for the
interval since the previous sample. Time in mode `manual` (human control) and `stopped` (for
example while the gate withholds motion) both reduce uptime.
Gate test: Mann-Whitney U test on per-mission uptime.

### `localisation_error_mean_m`

Unit: m. Lower is better.
Sum of `localisation_error_m` over all `state` samples divided by the number of samples. The log
format assumes that the robot (or an external reference) reports this error. The toolkit does not
estimate it.
Gate test: Mann-Whitney U test on per-mission mean errors.

### `faults_per_hour`

Unit: 1/h. Lower is better.
Number of `fault` events of any severity divided by total mission duration in hours.
Gate test: exact conditional test for two Poisson rates with mission hours as exposure.

## Gate verdicts

| Verdict | Condition |
| --- | --- |
| FAIL | Candidate violates `min_value` or `max_value`, or the regression exceeds `max_relative_regression` and the one-sided p-value is below `alpha`. |
| WARN | The regression exceeds the tolerance but is not significant (usually too few missions), or it is significant but within the tolerance. |
| PASS | Anything else. |

The overall verdict is FAIL if any gate fails, WARN if any gate warns, PASS otherwise. No
correction for multiple comparisons is applied.
