# Architecture

This document describes the reference design that the Bicep in `infra/` implements and that the
ADRs in `docs/adr/` justify. Four views are given: context, container, network and a request sequence.

Status of this design: the templates compile with the Bicep CLI and pass the `archkit` static
policy validator. They have not been deployed to an Azure subscription by the author of this
repository. Treat the design as a reviewed starting point, not as a certified landing zone.

## 1. Context

Who uses the platform and which external systems it depends on.

```mermaid
flowchart LR
    user([Employee or internal app])
    admin([Platform engineer])
    owner([Knowledge owner])

    subgraph platform[Agentic RAG platform]
        gw[AI gateway]
        orch[Agent orchestrator]
        kb[(Knowledge index)]
    end

    entra[Microsoft Entra ID]
    models[Azure OpenAI models in AI Foundry]
    sources[(Document sources)]
    monitor[Azure Monitor]

    user -- "question + Entra token" --> gw
    gw --> orch
    orch -- "hybrid query with ACL filter" --> kb
    orch -- "chat and embedding calls" --> models
    owner -- "publishes documents" --> sources
    sources -- "indexer pull" --> kb
    user -. "sign in" .-> entra
    gw -. "validate token" .-> entra
    admin -- "Bicep, pipelines, policy" --> platform
    platform -- "logs, metrics, traces" --> monitor
```

## 2. Container view

Deployable units and the Azure services behind them. Every arrow between services is authenticated
with a managed identity and a data plane role (ADR 0003).

```mermaid
flowchart TB
    client([Client])

    subgraph rg[Resource group per environment]
        apim["API Management<br/>internal VNet mode<br/>token validation, token limits, metrics"]

        subgraph cae[Container Apps environment, internal]
            orch["Orchestrator app<br/>planner, tool router, answer synthesis<br/>user-assigned identity"]
        end

        aif["AI Foundry account<br/>chat-small, chat-large, embedding<br/>local auth disabled"]
        search["Azure AI Search<br/>hybrid index, semantic ranker<br/>local auth disabled"]
        st[("Storage account<br/>source documents")]
        kv["Key Vault<br/>RBAC, purge protection"]
        law["Log Analytics +<br/>Application Insights"]
    end

    client -- HTTPS --> apim
    apim -- "backend with circuit breaker" --> orch
    orch -- "Cognitive Services OpenAI User" --> aif
    orch -- "Search Index Data Reader" --> search
    orch -- "Key Vault Secrets User" --> kv
    search -- "Storage Blob Data Reader (indexer)" --> st
    search -- "OpenAI User (integrated vectorization)" --> aif
    apim -. diagnostics .-> law
    orch -. traces .-> law
    aif -. diagnostics .-> law
    search -. diagnostics .-> law
    kv -. audit .-> law
```

| Container | Bicep module | Responsibility |
| --- | --- | --- |
| API Management | `modules/apim.bicep`, `policies/chat-api-policy.xml` | Single entry point, Entra token validation, tokens-per-minute limit per subscription, token metrics for showback |
| Orchestrator | `modules/container-apps.bicep` | Agent loop: plan, retrieve, call tools, synthesise a cited answer, route between small and large model |
| AI Foundry account | `modules/openai.bicep` | Model deployments with pinned versions and content filter policy |
| AI Search | `modules/search.bicep` | Hybrid (keyword + vector) index with security trimming fields |
| Storage | `modules/storage.bicep` | Source documents for the indexer |
| Key Vault | `modules/keyvault.bicep` | Third-party secrets and optional customer-managed keys |
| Network | `modules/network.bicep`, `modules/private-endpoint.bicep` | VNet, subnets, NSGs, private DNS zones, private endpoints |
| Identity | `modules/identity.bicep`, `modules/role-assignments.bicep` | Workload identity and least-privilege role assignments |
| Monitoring | `modules/monitoring.bicep` | Log Analytics workspace and Application Insights |

## 3. Network view

No PaaS data service has a public endpoint. The gateway is reachable only from inside the virtual
network (or from peered networks and an upstream Application Gateway, which are out of scope here).

```mermaid
flowchart TB
    corp(["Corporate network / hub VNet"])

    subgraph vnet["Spoke VNet 10.40.0.0/20"]
        subgraph s1["snet-apim 10.40.0.0/27 + NSG"]
            apim["API Management<br/>internal"]
        end
        subgraph s2["snet-container-apps 10.40.2.0/23 + NSG<br/>delegated to Microsoft.App/environments"]
            orch[Orchestrator replicas]
        end
        subgraph s3["snet-private-endpoints 10.40.1.0/26 + NSG"]
            pe1["PE: AI Foundry"]
            pe2["PE: AI Search"]
            pe3["PE: Key Vault"]
            pe4["PE: Blob"]
        end
        dns[["Private DNS zones<br/>openai, cognitiveservices, services.ai,<br/>search, vaultcore, blob"]]
    end

    aif["AI Foundry account<br/>publicNetworkAccess: Disabled"]
    search["AI Search<br/>publicNetworkAccess: disabled"]
    kv["Key Vault<br/>publicNetworkAccess: Disabled"]
    st[("Storage<br/>publicNetworkAccess: Disabled")]

    corp -- "peering / ExpressRoute, 443" --> apim
    apim -- "443" --> orch
    orch -- "443" --> pe1 & pe2 & pe3 & pe4
    pe1 --- aif
    pe2 --- search
    pe3 --- kv
    pe4 --- st
    dns -. "A records" .-> s3
```

Known exception: the Log Analytics workspace keeps public ingestion and query endpoints. Closing
them needs an Azure Monitor Private Link Scope, which is usually owned by the platform landing zone
and is therefore not created by this workload template (see ADR 0002).

## 4. Request sequence

One chat request, including the model routing decision (ADR 0004) and the spillover path from
provisioned to standard capacity.

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant G as API Management
    participant O as Orchestrator
    participant S as AI Search
    participant M1 as chat-small (PTU)
    participant M2 as chat-large (standard)
    participant L as Azure Monitor

    U->>G: POST /rag/chat (Bearer token)
    G->>G: validate Entra token, check token budget
    alt token budget exhausted
        G-->>U: 429 with Retry-After
    else within budget
        G->>O: forward with caller object id
        O->>M1: plan (managed identity token)
        M1-->>O: tool call: search(query)
        O->>S: hybrid query + security filter on caller groups
        S-->>O: ranked chunks with source ids
        O->>O: routing rule: complexity, context size, retries
        alt simple question
            O->>M1: synthesise answer from chunks
            M1-->>O: answer with citations
        else escalation or 429 from PTU
            O->>M2: synthesise answer from chunks
            M2-->>O: answer with citations
        end
        O->>O: check that each citation maps to a retrieved chunk
        O-->>G: answer, citations, token usage
        G->>L: token metrics by subscription
        G-->>U: 200 answer with citations
    end
```

## Design decisions

| ADR | Decision |
| --- | --- |
| [0001](adr/0001-buy-vs-build.md) | Use managed Azure AI services for models and retrieval, build only the orchestrator |
| [0002](adr/0002-private-networking.md) | Private endpoints and disabled public access for all data services |
| [0003](adr/0003-identity-and-zero-trust.md) | Managed identity and Entra ID everywhere, no keys |
| [0004](adr/0004-model-routing.md) | Small model by default, rule-based escalation to the large model |
| [0005](adr/0005-data-residency.md) | Regional or data zone deployments only, single-geography data flow |
| [0006](adr/0006-cost-governance.md) | Gateway token limits, showback tags and PTU sizing from measured demand |
| [0007](adr/0007-compute-platform.md) | Container Apps instead of AKS for the orchestrator |
| [0008](adr/0008-evaluation-and-responsible-ai.md) | Evaluation gate and Responsible AI controls as release criteria |

## How archkit relates to the design

```mermaid
flowchart LR
    w[workload.yaml] --> est[archkit estimate]
    p["illustrative_prices.yaml<br/>placeholders"] --> est
    est --> r1[Capacity and cost report]
    a[architecture.yaml] --> chk[archkit assess]
    rules[waf_rules.yaml] --> chk
    chk --> r2[Pillar scores, gaps, gate]
    b[infra/*.bicep] --> val[archkit validate-bicep]
    val --> r3[Policy findings, gate]
    r1 -. "sizes PTU, search SKU" .-> b
```
