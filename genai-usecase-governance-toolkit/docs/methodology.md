# Methodology

## Scoring

For each dimension d (value, feasibility, risk) with criteria c, weights w_c and scores s_c on the
scale [min, max]:

    dim_d = (sum(w_c * s_c) / sum(w_c) - min) / (max - min)        in 0..1

    overall = 100 * (W_value * dim_value + W_feas * dim_feas + W_risk * (1 - dim_risk)) / sum(W)

Quadrants use a threshold of 0.5 on value and feasibility: quick win (both high), strategic bet
(value high), fill-in (feasibility high), deprioritise (both low).

## Sensitivity

- One-at-a-time: a normalised dimension weight is shifted by delta, the other two are rescaled
  proportionally so that the weights sum to 1. Reported: Kendall tau against the baseline ranking,
  the largest rank shift of any use case and whether rank 1 changed.
- Monte Carlo: weights are drawn from Dirichlet(concentration * baseline weights). With the default
  concentration of 50 and a baseline weight of 0.45 the standard deviation of that weight is about
  0.07. Reported per use case: mean, best and worst rank, P(top k), P(same rank).
- Switch point: the smallest shift of a single weight, searched in steps of 0.005 in both
  directions, that changes the rank 1 use case.

## Build / buy / partner

Each factor rating r (1 to 5) is centred to (r - 3) / 2. The option score is the dot product with
the coefficient row of the option in `BUILD_BUY_MATRIX`. A margin below 0.5 between the best and the
second option is reported as a close call.

## Business case

Year 0 holds build and change management cost. In operating year t with adoption a_t:

    benefit_t   = annual_benefit_at_full_adoption * a_t
    inference_t = requests * (in_tokens / 1000 * price_in + out_tokens / 1000 * price_out) * a_t
    net_t       = benefit_t - run_cost - inference_t

NPV discounts net flows with the first flow at time zero. Payback is undiscounted with linear
interpolation inside the year. IRR is found by bisection on [-99%, 1000%]. ROI is
(total benefit - total cost) / total cost, undiscounted.

Uncertainty: each driver (benefit, build_cost, run_cost, volume, adoption) may declare low and high
multipliers. The tornado sets one driver to low and to high. The Monte Carlo simulation draws each
multiplier from a triangular distribution (low, 1.0, high), independently.

## Risk triage

Precedence: any Article 5 question answered yes gives "prohibited". Otherwise any Article 6
question gives "high-risk", unless only Annex III areas are touched, at least one Art. 6(3)
condition is met and there is no profiling of natural persons. Otherwise any Article 50 trigger
gives "limited". Otherwise "minimal". References used by the tool: Art. 4, 5, 6, 9 to 15, 17, 26,
27, 43, 47 to 50, 72, 73 and Annex I and III of Regulation (EU) 2024/1689. This is a simplified aid
and not legal advice.

## Fairness metrics

With selection rate SR_g = P(pred = 1 | group g), TPR_g and FPR_g per group:

    demographic parity difference = max_g SR_g - min_g SR_g
    equalised odds difference     = max(max_g TPR_g - min_g TPR_g, max_g FPR_g - min_g FPR_g)
    disparate impact ratio        = min_g SR_g / max_g SR_g      (four-fifths rule flags < 0.8)

Default flag thresholds are 0.10 for both differences and 0.80 for the ratio. Groups with fewer than
30 rows are flagged as unreliable. The bootstrap interval for the demographic parity difference
resamples rows with replacement and reports the 2.5 and 97.5 percentiles.

## Roadmap and risk register

Waves are assigned in rank order. Quick wins and fill-ins may start in wave 1, strategic bets and
high-risk use cases in wave 2 at the earliest, deprioritised use cases in the last wave. A use case
never starts before its dependencies. All waves except the last have a capacity limit. Prohibited
use cases and use cases that depend on them are excluded.

Risk register entries come from risk criteria scored 3 or higher (likelihood = score), the triage
tier and optional fairness flags. Impact is 1 + round(4 * value dimension). Severity is likelihood
times impact; 15 or more is high, 8 to 14 is medium.
