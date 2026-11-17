# ADR 0002: Private networking

- Status: accepted
- Deciders: security architect, network lead, solution architect
- Related: ADR 0003, ADR 0005

## Context

The platform processes confidential documents and prompts. Azure PaaS services are reachable over
public endpoints by default. The security baseline of the organisation requires that data services
are not reachable from the internet and that data exfiltration paths are limited.

## Options

1. Public endpoints with IP firewall rules. Simple, but allow-lists drift, shared egress IPs widen
   the trust boundary, and the control plane still resolves public names.
2. Service endpoints. Traffic stays on the backbone, but the service keeps a public address and
   on-premises clients cannot use them.
3. Private endpoints with public network access disabled. Each service gets a private IP in the
   spoke VNet and resolves through private DNS zones.

## Decision

Option 3 for every data service: AI Foundry account, AI Search, Key Vault and Storage.

- `publicNetworkAccess` is set to `Disabled` as a literal in the templates, not as a parameter, so
  that no environment can switch it on by configuration. Development uses the same posture.
- One private endpoint subnet with an NSG. Private DNS zones are created by the network module and
  linked to the spoke VNet. In a hub and spoke landing zone, replace them with the central zones.
- API Management runs in internal VNet mode. The Container Apps environment is internal and the
  orchestrator ingress is not external.
- AI Search reaches Storage and the AI account through its managed identity and the trusted
  service bypass. Where the bypass is not acceptable, use shared private links from AI Search.
- The static validator in `archkit` enforces the decision with rules NET-001 and NET-002.

## Consequences

Positive:

- No data service is addressable from the internet. Stolen credentials cannot be used from outside.
- The network posture of development and production is identical, so network defects surface early.

Negative and risks:

- Developers need a path into the VNet (VPN, Bastion with a jump host, or a dev box).
- Private DNS is a frequent source of deployment problems. Mitigation: zones and links are in code.
- Internal mode API Management needs the Developer or Premium tier, which raises the fixed cost.
- Exception: the Log Analytics workspace keeps public ingestion and query endpoints. Closing them
  needs an Azure Monitor Private Link Scope, which is shared across workloads and belongs to the
  landing zone. This exception is documented, not hidden.
- Outbound traffic from the Container Apps subnet is restricted by NSG only. A hub firewall with
  forced tunnelling is recommended and out of scope of this workload template.
