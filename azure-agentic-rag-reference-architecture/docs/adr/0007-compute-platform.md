# ADR 0007: Compute platform for the orchestrator

- Status: accepted
- Deciders: platform lead, lead engineer, solution architect
- Related: ADR 0001, ADR 0002

## Context

The orchestrator is a stateless HTTP service. It holds long-lived streaming connections, calls
models and search, and scales with concurrent requests. It needs VNet integration, managed
identity, revisions with traffic splitting and scale to a small footprint outside office hours.

## Options

| Criterion | Azure Container Apps | AKS | App Service |
| --- | --- | --- | --- |
| Operations effort | low, no cluster to run | high, cluster upgrades and add-ons | low |
| VNet integration, internal ingress | yes | yes | yes |
| Scale on concurrency, scale to low replica count | built in (KEDA) | to be configured | limited rules |
| Revisions and traffic splitting | built in | with a mesh or ingress controller | deployment slots |
| Sidecars, custom networking, GPU nodes | limited | full control | no |
| Fit for a small team | good | poor unless a cluster already exists | good |

## Decision

Azure Container Apps with an internal environment and workload profiles.

AKS is the better choice when the organisation already operates a hardened cluster platform, when
the workload needs GPU inference next to the orchestrator, or when a service mesh and custom
network policies are required. The orchestrator image and its configuration through environment
variables are portable to AKS without code changes.

## Consequences

Positive:

- No cluster lifecycle. Zone redundancy, mTLS between apps and scaling rules are configuration.
- Multiple revisions allow canary releases tied to the evaluation gate (ADR 0008).

Negative and risks:

- Less control over networking than AKS. Egress filtering depends on user-defined routes and a
  hub firewall.
- The delegated subnet must be sized up front. The template uses a /23.
- Cold starts if the minimum replica count is zero. Production keeps a minimum of three replicas.
