# ADR 0001: Buy versus build

- Status: accepted
- Deciders: solution architect, platform lead, security architect
- Related: ADR 0004, ADR 0006, ADR 0007

## Context

An enterprise wants a question answering assistant over internal documents that can also call
tools (an agentic RAG system). Three layers must be provided: language models, retrieval, and the
orchestration logic that plans, retrieves, calls tools and writes a cited answer. For each layer the
team can consume a managed service or operate its own stack.

Constraints: a small platform team (fewer than five engineers), an existing Azure landing zone,
confidential data, an EU residency requirement, and a need to change orchestration logic weekly.

## Options

| Layer | Option A: managed | Option B: self-hosted | Option C: packaged product |
| --- | --- | --- | --- |
| Models | Azure OpenAI deployments in AI Foundry | Open-weight models on GPU node pools | Included in the product |
| Retrieval | Azure AI Search | Open-source vector database on AKS | Included in the product |
| Orchestration | Custom service | Custom service | Low-code agent builder |

Assessment against the decision drivers:

| Driver | A: managed services + custom orchestrator | B: self-hosted stack | C: packaged product |
| --- | --- | --- | --- |
| Time to first production release | weeks | months (GPU capacity, serving, patching) | days to weeks |
| Operations load for a small team | low | high | lowest |
| Control over retrieval and agent logic | high | highest | limited |
| Private networking and Entra integration | native | to be built | depends on product |
| Unit cost at low and medium volume | pay per token, no idle GPUs | idle GPU cost dominates | per-seat or per-message |
| Exit cost | medium (API-compatible models, index is rebuildable) | low | high |

## Decision

Buy the commodity layers and build the differentiating one.

- Models: managed Azure OpenAI deployments. Self-hosting is reconsidered only if a required model
  is not offered, or if sustained volume makes dedicated GPUs cheaper than provisioned throughput.
- Retrieval: Azure AI Search with hybrid queries and the semantic ranker.
- Orchestration: a custom, containerised orchestrator service owned by the team. Agent logic,
  prompts and routing rules live in source control and are evaluated in CI (ADR 0008).

## Consequences

Positive:

- The team operates one stateless service instead of GPU clusters and a vector database.
- Private endpoints, managed identity and diagnostics are first-party features (ADR 0002, 0003).
- The orchestrator talks to models through a narrow interface, so a model or provider can be swapped.

Negative and risks:

- Dependence on regional model availability and quota. Mitigation: spillover deployments and the
  routing fallback in ADR 0004.
- Token-based pricing must be governed actively (ADR 0006).
- AI Search index schema and scoring are less tunable than a self-managed engine.

Revisit when: monthly model spend exceeds the cost of dedicated capacity by a clear margin for
three consecutive months, or a compliance requirement rules out the managed service.
