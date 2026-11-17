# ADR 0008: Evaluation and Responsible AI

- Status: accepted
- Deciders: product owner, lead engineer, risk officer, solution architect
- Related: ADR 0003, ADR 0004

## Context

A RAG assistant can fail in ways that infrastructure monitoring does not see: it can retrieve the
wrong passages, state something the sources do not support, follow instructions hidden in a
document, or reveal personal data. Prompts, models, chunking and index settings change often, and
every change can move quality in either direction.

## Decision

Quality and safety are release criteria, measured the same way on every change.

Evaluation:

- A versioned evaluation set with questions, expected source documents and reference answers,
  owned by the product owner and extended from user feedback.
- Offline metrics on every pull request that touches prompts, routing, chunking or models:
  retrieval recall and ranking quality, groundedness of answers, citation correctness, refusal
  behaviour on unanswerable questions, latency percentiles and tokens per request.
- A gate in CI: a release is blocked if a metric falls below its threshold or regresses by more
  than an agreed margin against the last release.
- Online: sampled traces are scored for groundedness, and user feedback is linked to the trace id.

Safeguards:

- Content filters on every deployment (`raiPolicyName` in the model deployment). They are not
  switched off per request.
- Prompt injection: user input and retrieved passages are screened, retrieved text is placed in a
  clearly delimited data section, and tool calls are restricted to an allowlist with typed
  arguments. Tools that change state need user confirmation (ADR 0003).
- Answers cite sources. If no passage supports an answer, the assistant says so.
- Personal data is redacted before content is written to logs or evaluation stores.
- Human oversight: a feedback control in every client, an escalation route to a named owner, and a
  documented incident process for harmful or wrong output.
- Transparency: users are told that they interact with an AI system, what it is for and what its
  limits are.
- Red teaming before go-live and after major changes.

## Scope of this repository

The Bicep here deploys the content filter policy reference and the telemetry destination. The
evaluation gate and the safeguards above are application and process controls. They are listed
in the checklist rules (RAI-01 to RAI-08) so that an architecture review asks for them, but this
repository does not implement them. The example description declares `red_teaming: false`.

## Consequences

Positive:

- Changes are compared on evidence. Model upgrades become routine releases.
- Reviewers and auditors get a fixed list of controls with a status each.

Negative and risks:

- The evaluation set needs continuous curation, otherwise the gate measures the past.
- Automated groundedness scoring is itself a model output and must be spot-checked by people.
- Safeguards add latency and tokens. The cost is accepted and included in the estimate inputs.
