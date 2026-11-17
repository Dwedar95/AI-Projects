# ADR 0003: Identity and zero trust

- Status: accepted
- Deciders: security architect, identity lead, solution architect
- Related: ADR 0002, ADR 0006

## Context

Network isolation alone is not sufficient: a compromised workload inside the VNet could still read
every index and call every model if services accept shared keys. Users also have different access
rights to the underlying documents, and an assistant must not become a way around those rights.

## Decision

Verify explicitly, use least privilege, assume breach.

Service to service:

- Every service call uses a managed identity and an Entra ID token. Key authentication is disabled:
  `disableLocalAuth: true` on the AI account and AI Search, `allowSharedKeyAccess: false` on
  Storage, `enableRbacAuthorization: true` on Key Vault.
- The orchestrator has one user-assigned identity with data plane roles only:
  Cognitive Services OpenAI User, Search Index Data Reader, Key Vault Secrets User.
- AI Search uses its system-assigned identity to read blobs and to call the embedding deployment.
- No workload identity holds Owner, Contributor or User Access Administrator. The validator rule
  IAM-003 rejects such role assignments.

User to platform:

- Callers present an Entra ID access token for the API audience. API Management validates issuer
  tenant and audience before the request reaches the orchestrator.
- An API Management subscription identifies the consuming application for quota and showback. It
  is an identifier for metering, not the security boundary.

Document-level authorization:

- Each chunk in the index carries the object ids of the Entra groups allowed to read its source.
- The orchestrator builds a security filter from the group claims of the caller and adds it to
  every query. The model never sees a chunk the caller is not allowed to read.
- Agent tools run with the least privilege needed. Tools that change state require explicit user
  confirmation and act on behalf of the user, not with the workload identity.

## Options considered and rejected

- API keys in Key Vault: still a shared secret, rotation burden, no per-caller audit trail.
- One system-assigned identity per app: rejected in favour of a user-assigned identity so that role
  assignments exist before the app revision starts and survive app recreation.
- Authorization only in the prompt ("do not reveal restricted content"): not a control.

## Consequences

Positive:

- No secrets to rotate for platform services. Access is auditable per identity in Entra sign-in logs.
- A leaked token is short-lived and scoped to one audience.

Negative and risks:

- Group claims can overflow the token (more than 200 groups). Mitigation: resolve groups through
  Microsoft Graph and cache them, or use application roles.
- ACL metadata must be kept in sync by the ingestion pipeline. Stale ACLs are a data leak. This is
  the highest-risk part of the design and needs its own tests.
- Local development needs developer identities with the same data plane roles in a dev environment.
