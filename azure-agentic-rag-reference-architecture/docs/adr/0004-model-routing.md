# ADR 0004: Model routing

- Status: accepted
- Deciders: solution architect, lead engineer, product owner
- Related: ADR 0006, ADR 0008

## Context

Large models give better multi-step reasoning but cost and latency per token are a multiple of a
small model. In enterprise assistants a large share of questions are simple lookups. Sending every
request to the largest model wastes budget, and sending every request to the smallest one lowers
answer quality on the hard cases.

## Options

1. Single large model for everything.
2. Single small model for everything.
3. Static routing by endpoint or use case.
4. Rule-based routing inside the orchestrator with escalation.
5. Learned router (classifier or a model that predicts the required model).

## Decision

Option 4, with the interface prepared for option 5.

- Default path: `chat-small` handles planning, tool selection and answer synthesis.
- Escalate to `chat-large` when one of these rules fires:
  - the plan contains more than a configured number of tool calls or a comparison across sources,
  - the retrieved context exceeds a token threshold,
  - the small model returned an answer that failed the citation check and one retry also failed,
  - the consuming application asked for the high-quality tier and is entitled to it.
- Capacity fallback is separate from quality escalation: on HTTP 429 or 5xx from the provisioned
  deployment, the same model class is called on the standard deployment (spillover).
- Embeddings always use the single embedding deployment that built the index. Changing it means a
  full re-index.
- Model versions are pinned (`versionUpgradeOption: NoAutoUpgrade`). A version change is a release
  that must pass the evaluation gate (ADR 0008).
- Every response records the model class and the routing reason, so the escalation rate is a metric.

## Consequences

Positive:

- Cost follows difficulty. The estimator in `archkit` takes the routing shares as input, so the
  financial effect of a routing change can be estimated before release.
- Rules are transparent and testable. Each rule has unit tests and an entry in the evaluation set.

Negative and risks:

- Rules need tuning. An escalation rate that creeps up erodes the saving. Mitigation: alert on it.
- Two model classes double the evaluation matrix.
- A learned router could do better. It is deferred until there is labelled traffic to train it on.
