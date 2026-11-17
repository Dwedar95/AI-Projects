# Playbook: augment or rewrite a workflow

This playbook describes how to decide what to do with a business process after a pilot of the
agent-augmented workflow. Every rule refers to a metric that `process-copilot simulate` (module
`analytics.py`) computes from case traces. In a real pilot the same metrics come from production
traces. The numbers quoted below come from the simulated case logs of this repository and only
illustrate how the rules are applied.

## The four outcomes

| Outcome | Meaning |
|---|---|
| Augment | Keep the process shape, roles and checkpoints. Let the workflow do the preparation steps (classify, look up, retrieve, summarise, draft) and leave decisions and approvals with people. |
| Rewrite | Redesign the process as AI-first. The workflow owns the standard case end to end, people handle exceptions and audits. Roles, SLAs and controls are redefined. |
| Stabilise | Do not scale yet. Too many cases fall out of the workflow. Fix the main failure point, then measure again. |
| Stop | The workflow does not remove enough work. Do not invest further in this process. |

## Metrics

| Metric | Definition in the code | Why it matters |
|---|---|---|
| Automation rate | Share of cases completed with zero interventions (`CaseTrace.automated`) | Shows whether the standard case can run without people |
| Intervention rate | Share of cases with at least one approval, manual step or handoff | The human workload that remains |
| Handoff rate | Share of cases the workflow gave up on | Quality signal: the workflow did not understand or was not allowed |
| Approval rate | Share of cases that waited for an approval | Shows whether control thresholds fit the case mix |
| Handling time saved | (minutes before - minutes after) / minutes before, paired per case | The benefit |
| Cost per case | Labour minutes at the hourly rate plus model cost | The business case |
| SLA breach rate | Share of cases where a step or the case exceeded its SLA | Guardrail |
| Handoffs by step | Count of handoffs per step id | Where to work first |

## Decision rules

The function `recommend` applies these rules in order. Thresholds are constants in `analytics.py`.
They are starting values and should be set per organisation.

1. **Handoff rate above 25 percent: stabilise.** A workflow that gives up on more than a quarter of
   the cases creates rework and loses trust. Look at "handoffs by step" and fix the largest one.
2. **Handling time saved below 20 percent: stop.** The remaining manual steps dominate. Automating
   the preparation does not pay for run cost and change effort.
3. **Automation rate at least 60 percent and handoff rate at most 10 percent: rewrite.** The
   standard case already runs without people and exceptions are rare. Keeping the old process
   shape, with its queues and review steps designed for manual work, now costs more than it protects.
4. **Otherwise: augment.**

Two notes are added independently of the outcome:

- **Approval rate above 30 percent:** review the approval thresholds. Either the threshold is lower
  than the risk requires, or the case mix has shifted.
- **SLA breach rate above 5 percent:** check the manual steps that remain, they are the bottleneck.

## What "rewrite" requires beyond the metrics

The metrics are necessary, not sufficient. Before a rewrite, check:

- **Reversibility.** Can a wrong automated action be undone (a refund, a payment, a message)? If
  not, keep an approval for it, whatever the automation rate says.
- **Regulatory exposure.** Decisions about people (HR, credit, benefits) may require human
  oversight by law or by policy. Such steps stay `human` steps.
- **Data classification.** Confidential steps are routed to self-hosted models in this reference.
  A rewrite must not weaken that.
- **Stable inputs.** A high automation rate measured on four weeks of cases is not stable if the
  product, the policy or the channel mix is about to change.
- **Ownership.** Someone owns the workflow, its prompts, its knowledge base and its metrics.

## Applied to the three example processes

Simulated case logs, 300 cases each, seed 7, mock models, fictional data, labour rate 45 USD per
hour (assumption). Reproduce with `make demo`.

| Process | Automation | Intervention | Handoff | Approval | Time saved | Recommendation |
|---|---|---|---|---|---|---|
| Customer care refund | 42.7% | 57.3% | 30.7% | 30.7% | 59.9% | Stabilise |
| Invoice exception handling | 49.0% | 51.0% | 3.3% | 28.3% | 61.4% | Augment |
| HR onboarding questions | 77.0% | 23.0% | 15.0% | 0.0% | 70.6% | Augment |

**Refund: stabilise.** 55 of the 92 handoffs come from `issue_refund`: the simulated team lead
denies refunds above 500 USD and the case goes to manual review. That is a policy choice, not a
model failure. The first fix is to route high-value refunds to a dedicated review step instead of
treating the denial as a failure, the second is to ask the customer for a missing order id
(`extract_details`, 23 handoffs) instead of handing off. The approval rate of 30.7 percent
triggers the threshold review note.

**Invoice exceptions: augment.** Clean invoices run straight through (49 percent). Exceptions always
end in a manual step, by design: resolving a price variance needs a conversation with the supplier.
The workflow removes the preparation (classification, policy lookup, summary, ticket), which is
where the 61.4 percent time saving comes from. A rewrite is not indicated because half of the cases
need a person.

**HR onboarding: augment, close to rewrite.** 77 percent of the questions are answered from the
handbook. The handoff rate of 15 percent is above the rewrite limit of 10 percent. All handoffs are
at `classify_topic`: questions outside the handbook and questions that mix payroll with a dispute.
Sensitive topics go to an HR partner as a `human` step and must stay there after any redesign.
Extending the handbook and the topic list is the way to reach the rewrite thresholds.

## How to use this in a pilot

1. Model the current process in YAML with honest `manual_minutes` per step (time study or system logs).
2. Run the pilot with tracing on and collect at least a few hundred cases.
3. Compute the metrics, read "handoffs by step" first.
4. Apply the rules, then the checklist above.
5. Change one thing (a prompt, a threshold, a knowledge article), measure again, compare.
