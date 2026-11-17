# ADR 0005: Data residency

- Status: accepted
- Deciders: data protection officer, security architect, solution architect
- Related: ADR 0002, ADR 0004

## Context

The reference customer is subject to EU data protection law and has a policy that confidential
data is stored and processed within the EU. Azure OpenAI offers deployment types with different
processing locations: global (any Azure region), data zone (within a defined zone such as the EU)
and regional (one region). Capacity and model availability are best for global and worst for
regional deployments.

This record is an architecture decision. It is not legal advice, and the deployment type
definitions and availability must be checked against current Microsoft documentation and the
customer's own legal assessment before adoption.

## Options

| Option | Processing location | Capacity and model choice | Fit with an EU-only policy |
| --- | --- | --- | --- |
| Global deployments | Any region | Best | Does not fit |
| Data zone deployments | Within the EU data zone | Good | Fits if the policy is EU-wide |
| Regional deployments | One region | Most constrained | Fits, also for single-country policies |

## Decision

- All resources that store data (Storage, AI Search, Key Vault, Log Analytics) are deployed in one
  EU region chosen by the `location` parameter.
- Chat model deployments use data zone SKUs in production to gain capacity while staying inside
  the EU. Regional SKUs are used where a customer policy names a country or region.
- Global deployment SKUs are not allowed. The checklist rule SEC-06 fails an architecture that
  declares a residency requirement and uses a global deployment.
- Prompts and completions are not written to logs in clear text. Diagnostic logs carry metadata
  (token counts, latency, model, caller id). Where content logging is needed for evaluation, it
  goes to a separate store in the same region with its own retention and access roles.
- Disaster recovery, if added, uses a second region in the same geography.

## Consequences

Positive:

- A single statement can be made about the data flow: storage in one region, processing in the EU.
- The rule is enforceable in review tooling rather than left to convention.

Negative and risks:

- New models often arrive on global deployments first. The platform may lag by weeks or months.
- Regional quota can be tight. Mitigation: request quota early, use spillover within the zone.
- Features that process data outside the chosen boundary must be reviewed one by one before they
  are enabled. The default is off.
