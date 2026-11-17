# ADR 0006: Cost governance

- Status: accepted
- Deciders: platform lead, finance partner, solution architect
- Related: ADR 0001, ADR 0004

## Context

Model spend scales with tokens, and agentic workloads multiply tokens per user request because
each request triggers several model calls with growing context. Without guardrails one consuming
application or one runaway agent loop can consume the shared quota and the monthly budget.

## Decision

Govern cost at four points.

1. Attribution. Every resource carries `costCenter`, `owner` and `environment` tags (validator
   rules GOV-001 and GOV-002). The gateway emits token metrics with the API Management
   subscription as a dimension, which gives showback per consuming application.
2. Limits. The gateway enforces a tokens-per-minute limit per subscription
   (`llm-token-limit` policy). The orchestrator enforces a maximum number of agent iterations and
   a token budget per request. Log Analytics has a daily cap outside production.
3. Capacity choice. Provisioned throughput (PTU) is bought only for the base load that keeps its
   utilization above the break-even, and peaks spill over to pay-as-you-go. The sizing comes from
   `archkit estimate` with measured traffic and current prices, and is reviewed monthly.
4. Budgets. One budget per environment with alerts at 50, 80 and 100 percent of the forecast,
   routed to the owner tag. Anomaly alerts on daily token volume per subscription.

Design rules that reduce unit cost:

- Route to the small model by default (ADR 0004).
- Keep the system prompt and tool definitions stable and at the start of the prompt so that the
  prefix can be served from the prompt cache.
- Retrieve fewer, better chunks: hybrid retrieval with reranking and a cap on context tokens.
- Use scalar quantization for vectors when the recall loss measured on the evaluation set is
  acceptable. It reduces the vector index size and can avoid a larger search tier.

## About the numbers in this repository

`src/archkit/data/illustrative_prices.yaml` contains placeholder prices and capacities. They are
not Azure list prices. The estimator is a model of the arithmetic, and its output is only as good
as the price book and the traffic assumptions given to it.

## Consequences

Positive:

- Spend is attributable, bounded per consumer and reviewed on a schedule.
- The PTU decision is reproducible: inputs and formulae are in the repository.

Negative and risks:

- Token limits can reject legitimate peaks. Mitigation: per-product limits and a documented
  process to raise them.
- PTU reservations are a commitment. A wrong forecast costs money in either direction.
- Showback by subscription needs discipline: one subscription per consuming application.
