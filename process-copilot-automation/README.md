# process-copilot-automation

A reference implementation of AI-first process automation: a workflow engine that turns a business
process defined in YAML into an agent-augmented workflow, with model routing, policy-based human
approval, per-step tracing and adoption and ROI analytics.

## What it is

You describe a process as steps with roles, systems, SLAs and manual handling times. The engine
executes it with five step types:

- `prompt`: an LLM task (classify, extract, summarise, generate). Outputs feed later steps (prompt chaining).
- `rag`: lookup in a small knowledge base with a TF-IDF retriever, then a grounded answer with sources.
- `action`: a call to a registered internal API tool (mock CRM, finance, ticketing, HR), restricted
  by an allowlist and gated by approval rules.
- `decision`: a branch on a condition over the workflow context.
- `human`: a step that stays manual.

Around the engine: a model router (cheap versus strong model by task complexity and case budget,
proprietary versus self-hosted open-source hosting by data classification), a provider abstraction
with a deterministic offline mock, traces with latency, tokens, cost and human time per step, and an
analytics module that computes automation rate, handling time saved, cost per case, intervention
rate and a before / after comparison from simulated case logs.

Three example processes are included: customer care refund, invoice exception handling and HR
onboarding questions. `docs/playbook.md` describes when to augment and when to rewrite a workflow,
using the metrics the code produces.

## Background

Reference implementation by the author of the architecture pattern it shows: M. Emad Dwedar, AI
researcher at TUM Heilbronn, co-founder of AsetraAI, Microsoft AI Consultant and Azure Solutions
Architect Expert. It condenses how the author designs agent-augmented business workflows: process
as data, tools behind an allowlist, approvals as policy, routing by cost and data class, and
decisions about scaling based on measured adoption.

This is an independent open-source reference. It is not production code of any company and not a
product of Microsoft or any other vendor. All companies, policies, records and case logs in this
repository are fictional or simulated. No real language model was called to produce the output
shown below.

## Features

| Area | Module | Notes |
|---|---|---|
| Process model | `process.py` | pydantic validation of steps, branches, allowlist, approval rules |
| Engine | `engine.py` | context passing, branching, fail-closed handoff with reason |
| Conditions and templates | `expressions.py` | `{{ path }}` rendering, AST-based restricted evaluator, no `eval` |
| Retrieval | `retriever.py` | TF-IDF with sublinear tf, smoothed idf, cosine similarity, paragraph chunking |
| Providers | `providers.py` | `LLMProvider` protocol, deterministic `MockProvider`, optional OpenAI-compatible adapter |
| Routing | `routing.py` | complexity score, budget downgrade, hosting by data classification, cascade on low confidence |
| Tools | `tools.py` | registry with argument checking, mock CRM / finance / ticketing / HR with seeded fictional data |
| Approval | `policy.py` | first-match rules, pluggable approver, denied when no approver is configured |
| Observability | `tracing.py` | per-step model, tokens, cost, latency, confidence, approval, human minutes, SLA |
| Analytics | `analytics.py`, `simulation.py` | paired before / after, ROI projection, playbook recommendation |
| Interfaces | `cli.py`, `api.py` | `process-copilot` CLI, optional FastAPI app |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

process-copilot validate examples/processes/customer_refund.yaml
process-copilot run examples/processes/customer_refund.yaml \
    --input customer_id=C-1007 --input "message=Order ORD-1000 was faulty and I want it refunded."
process-copilot run examples/processes/invoice_exception.yaml --input invoice_id=INV-2000 --json
process-copilot simulate examples/processes/hr_onboarding.yaml --cases 300 --seed 7
process-copilot tools
```

Optional HTTP API (needs `pip install -e ".[serve]"`):

```bash
uvicorn process_copilot.api:create_app --factory
curl -X POST localhost:8000/processes/hr_onboarding/run -H "Content-Type: application/json" \
    -d '{"inputs": {"employee_id": "E-3001", "question": "When is the payroll date?"}}'
```

Using real models (untested, see limitations): set `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`
(or `OPENAI_BASE_URL`, `OPENAI_API_KEY` for OpenAI or a self-hosted OpenAI-compatible server) and pass
`providers={"proprietary": OpenAICompatibleProvider.from_env(), "open_source": ...}` plus a
`ModelRouter` with your own catalog to `WorkflowEngine`.

## Demo output

Pasted from a real run of `python examples/run_demo.py` in this repository (Python 3.13). The models
are the deterministic mock, so model latency is computed from a formula and not measured, token
counts are word and punctuation counts, and prices are illustrative catalog values, not vendor prices.

Single cases with traces. `prop-large*` marks a step that was escalated from the cheap to the
strong model after low confidence:

```text
=== 1. Validate a process ===
$ process-copilot validate examples/processes/customer_refund.yaml

OK: customer_refund (Customer care refund request)
  steps: 11 (3 action, 2 decision, 1 human, 4 prompt, 1 rag)
  roles: Service agent
  systems: CRM, Finance, Knowledge base, Ticketing
  tools allowed: crm.get_order, finance.issue_refund, ticketing.create_ticket
  approval rules: 1, case budget: 0.02 USD, SLA: 30.0 min

=== 2. Refund with cascade and approval ===
$ process-copilot run examples/processes/customer_refund.yaml --input customer_id=C-1007 --input "message=Order ORD-1000 was faulty and I want it refunded."

Process: Customer care refund request | case case-1 | status: COMPLETED
step                type      status  model        tok in/out  cost usd  latency ms  conf  human min  approval / detail
------------------  --------  ------  -----------  ----------  --------  ----------  ----  ---------  ----------------------------------------------------------------------
classify_intent     prompt    ok      prop-large*  42/18       0.00015   766         1.00  -          {"label": "refund_request"}
route_intent        decision  ok                   -           -         -           -     -          intent.label == 'refund_request' -> extract_details
extract_details     prompt    ok      prop-small   21/11       0.00001   216         1.00  -          {"order_id": "ORD-1000"}
lookup_order        action    ok                   -           -         -           -     -          crm.get_order(order_id=ORD-1000)
check_policy        rag       ok      prop-small   123/24      0.00003   294         0.86  -          retrieved refund_policy#1, refund_policy#4, refund_policy#3
check_eligibility   decision  ok                   -           -         -           -     -          order.found and order.status == 'delivered' and order.days_since_deliv
issue_refund        action    ok                   -           -         -           -     2.0        approved by simulated reviewer (Team lead customer care); finance.issu
draft_confirmation  prompt    ok      prop-large   42/36       0.00046   1048        0.90  -          [mock draft] Your refund RF-0001 of 279.89 USD for order ORD-1000 has
close_ticket        action    ok                   -           -         -           -     -          not required; ticketing.create_ticket(queue=customer-care, subject=Ref

Totals: 317 tokens, model cost 0.00066 USD (budget 0.02 USD), model latency 2324 ms (simulated by the mock), human time 2.0 min vs 10.0 min manual, interventions 1, SLA breached: no

=== 3. Refund above the reviewer's limit ===
$ process-copilot run examples/processes/customer_refund.yaml --input customer_id=C-1007 --input "message=Please refund order ORD-1001, it is not what I ordered."

Process: Customer care refund request | case case-1 | status: HANDOFF (issue_refund: approval denied by simulated reviewer (Team lead customer care))
step               type      status  model       tok in/out  cost usd  latency ms  conf  human min  approval / detail
-----------------  --------  ------  ----------  ----------  --------  ----------  ----  ---------  ----------------------------------------------------------------------
classify_intent    prompt    ok      prop-small  23/9        0.00001   204         1.00  -          {"label": "refund_request"}
route_intent       decision  ok                  -           -         -           -     -          intent.label == 'refund_request' -> extract_details
extract_details    prompt    ok      prop-small  23/11       0.00001   216         1.00  -          {"order_id": "ORD-1001"}
lookup_order       action    ok                  -           -         -           -     -          crm.get_order(order_id=ORD-1001)
check_policy       rag       ok      prop-small  123/24      0.00003   294         0.86  -          retrieved refund_policy#1, refund_policy#4, refund_policy#3
check_eligibility  decision  ok                  -           -         -           -     -          order.found and order.status == 'delivered' and order.days_since_deliv
issue_refund       action    denied              -           -         -           -     11.0       denied by simulated reviewer (Team lead customer care); approval denie

Totals: 213 tokens, model cost 0.00005 USD (budget 0.02 USD), model latency 714 ms (simulated by the mock), human time 11.0 min vs 16.5 min manual, interventions 1, SLA breached: yes

=== 4. Invoice exception ===
$ process-copilot run examples/processes/invoice_exception.yaml --input invoice_id=INV-2000

Process: Invoice exception handling | case case-1 | status: COMPLETED
step                     type      status  model      tok in/out  cost usd  latency ms  conf  human min  approval / detail
-----------------------  --------  ------  ---------  ----------  --------  ----------  ----  ---------  ----------------------------------------------------------------------
get_invoice              action    ok                 -           -         -           -     -          finance.get_invoice(invoice_id=INV-2000)
check_po_reference       decision  ok                 -           -         -           -     -          invoice.found and invoice.po_id != '' -> get_po
get_po                   action    ok                 -           -         -           -     -          finance.get_purchase_order(po_id=PO-5000)
match_amounts            decision  ok                 -           -         -           -     -          default -> classify_exception
classify_exception       prompt    ok      oss-small  22/9        0.00000   290         1.00  -          {"label": "price_variance"}
lookup_policy            rag       ok      oss-small  125/30      0.00002   500         0.60  -          retrieved invoice_policy#3, invoice_policy#4, invoice_policy#1
summarise_case           prompt    ok      oss-large  73/30       0.00009   1350        0.90  -          Invoice INV-2000 from Supplier 1 over 4994.44 USD is blocked as price_
open_procurement_ticket  action    ok                 -           -         -           -     -          not required; ticketing.create_ticket(queue=procurement, subject=Invoi
resolve_exception        human     ok                 -           -         -           -     8.0        manual step by AP clerk

Totals: 289 tokens, model cost 0.00011 USD (budget 0.02 USD), model latency 2140 ms (simulated by the mock), human time 8.0 min vs 21.5 min manual, interventions 1, SLA breached: no
```

Simulated adoption analytics (300 simulated cases per process, seed 7). Full output for the refund
process, key lines for the other two:

```text
=== 6. Simulated adoption analytics: refund ===
$ process-copilot simulate examples/processes/customer_refund.yaml

Process: Customer care refund request
SIMULATED case log: 300 cases, seed 7, mock models, fictional system data, labour rate 45 USD/h (assumption)

metric                              before (manual)  after (augmented)
----------------------------------  ---------------  -----------------
handling time per case (min)                  10.89               4.36
labour + model cost per case (USD)            8.167              3.272
model cost per case (USD)                   0.00000            0.00034
cases without human involvement                0.0%              42.7%
cases with human intervention                100.0%              57.3%

automation rate:          42.7%
intervention rate:        57.3% (0.57 interventions per case)
handoff rate:             30.7%   approval rate: 30.7%
handling time saved:      59.9%
saving per case:          4.895 USD
avg tokens per case:      221   avg model latency: 1314 ms (simulated)
SLA breach rate:          23.0%
handoffs by step:         issue_refund=55, extract_details=23, classify_intent=14

ROI projection at 4000 cases per month (assumed volume and costs):
  gross saving 19,582 USD/month, platform cost 1,500, net 18,082 USD/month
  hours freed 435 per month, one-off cost 60,000 USD, payback 3.3 months

Playbook recommendation: STABILISE
  - handoff rate 31% is above 25%; fix the main failure point first (step 'issue_refund')
  - 31% of cases wait for an approval; review the approval thresholds
  - SLA breach rate 23%; check the manual steps that remain

=== 7. Simulated adoption analytics: invoice ===
SIMULATED case log: 300 cases, seed 7, mock models, fictional system data, labour rate 45 USD/h (assumption)
automation rate:          49.0%
intervention rate:        51.0% (0.51 interventions per case)
handoff rate:             3.3%   approval rate: 28.3%
handling time saved:      61.4%
saving per case:          3.834 USD
handoffs by step:         approve_payment=10
  hours freed 341 per month, one-off cost 60,000 USD, payback 4.3 months
Playbook recommendation: AUGMENT
  - automation rate 49% with intervention rate 51%; keep the process shape and the human checkpoints, automate the preparation steps
=== 8. Simulated adoption analytics: HR ===
SIMULATED case log: 300 cases, seed 7, mock models, fictional system data, labour rate 45 USD/h (assumption)
automation rate:          77.0%
intervention rate:        23.0% (0.23 interventions per case)
handoff rate:             15.0%   approval rate: 0.0%
handling time saved:      70.6%
saving per case:          4.171 USD
handoffs by step:         classify_topic=45
  hours freed 371 per month, one-off cost 60,000 USD, payback 4.0 months
Playbook recommendation: AUGMENT
  - automation rate 77% with intervention rate 23%; keep the process shape and the human checkpoints, automate the preparation steps
```

Reading: the refund workflow saves 59.9 percent handling time in the simulation but hands off 30.7
percent of the cases, mostly because the simulated reviewer denies refunds above 500 USD, so the
playbook says to stabilise before scaling. See `docs/playbook.md` for the rules and the discussion
of all three processes.

## Architecture

See [docs/architecture.md](docs/architecture.md) (mermaid diagram, step types, design decisions) and
[docs/playbook.md](docs/playbook.md) (augment versus rewrite).

```
src/process_copilot/
  process.py      process, step and approval rule models, YAML loader
  expressions.py  templates and restricted condition evaluator
  engine.py       workflow engine
  providers.py    provider interface, mock, optional HTTP adapter
  routing.py      model catalog and router
  retriever.py    TF-IDF retriever and knowledge base loader
  tools.py        tool registry and mock systems
  policy.py       approval policy and approvers
  tracing.py      step and case traces
  simulation.py   seeded case generators and simulated reviewer
  analytics.py    adoption metrics, ROI projection, recommendation
  cli.py, api.py  command line and FastAPI interfaces
examples/
  processes/      customer_refund.yaml, invoice_exception.yaml, hr_onboarding.yaml
  knowledge/      fictional policies used by the RAG steps
```

## Tests

```bash
make test   # pytest -q
make lint   # ruff check src tests examples
```

54 tests. They cover the expression evaluator including rejected unsafe input, process validation,
retrieval ranking, the mock tiers, routing rules and budget handling, every path of the three
example processes with their side effects in the mock systems, fail-closed behaviour, analytics
against hand-computed values, the CLI and the API.

## Limitations

- The mock provider is not a language model. It uses keyword matching, regular expressions and
  sentence selection. Results say nothing about the accuracy of real models on these tasks. The
  cheap and strong tiers differ by rules written for this demo.
- `OpenAICompatibleProvider` is untested against live endpoints. Only its message building and
  response parsing are unit-tested. Real models do not return calibrated confidence, so the
  confidence gates need a calibration method that this repository does not provide.
- Case logs are simulated. The case mix, the manual minutes per step, the labour rate, the model
  prices, the monthly volume and the platform cost are assumptions. The ROI figures show how the
  calculation works, they are not evidence of savings in a real organisation.
- Latency in traces is a formula of the mock (base plus per-token time). Wall-clock time of the
  engine is recorded separately per step. Queue and waiting time for approvals are not modelled,
  so SLA figures cover handling time only.
- The engine runs one case synchronously in memory. There is no persistence, no retry or
  compensation of tool calls, no parallel steps, no pause and resume of a case that waits for
  approval, and no authentication on the demo API.
- The TF-IDF retriever is lexical. It misses paraphrases that share no terms with the knowledge base.
- The mock systems are in-process Python objects, not network services.

## License

MIT, see [LICENSE](LICENSE). Copyright (c) 2026 M. Emad Dwedar.
