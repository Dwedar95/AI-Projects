"""Simulated case generators for the three example processes.

Everything produced here is SIMULATED. Cases are drawn from fixed templates and
the fictional mock system records with a seeded random generator. The mix of
case types is an assumption chosen to exercise every path of the workflows.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

from .policy import ApprovalRequest, CallbackApprover
from .tools import MockSystems

REFUND_TEMPLATES: tuple[tuple[float, str], ...] = (
    (0.34, "I would like a refund for order {oid}. The item arrived damaged."),
    (0.18, "Please refund order {oid}, it is not what I ordered."),
    (0.14, "Order {oid} was faulty and I want it refunded."),
    (0.08, "I want my money back for {oid}, the product stopped working."),
    (0.07, "I want a refund for my last order, it broke after a week."),
    (0.09, "Where is my order {oid}? The tracking has not changed for days."),
    (0.06, "I am very unhappy with your support, this is a complaint."),
    (0.04, "Hello, I have a question about my account."),
)

HR_QUESTIONS: tuple[tuple[float, str], ...] = (
    (0.22, "When do I get my laptop and VPN access?"),
    (0.20, "How many vacation days do I have in my first year?"),
    (0.16, "When is the payroll date and when will I get paid?"),
    (0.14, "How do I enrol in the pension and health insurance benefits?"),
    (0.08, "Who resets my password for the email account?"),
    (0.08, "I need to report harassment by a colleague."),
    (0.06, "My payslip seems wrong and I want to raise a salary dispute."),
    (0.06, "Can I bring my dog to the office?"),
)


def _pick(rng: random.Random, weighted: tuple[tuple[float, str], ...]) -> str:
    return rng.choices([t for _, t in weighted], weights=[w for w, _ in weighted], k=1)[0]


def refund_case(rng: random.Random, systems: MockSystems) -> dict[str, Any]:
    oid = rng.choice(sorted(systems.orders))
    return {"customer_id": systems.orders[oid]["customer_id"], "message": _pick(rng, REFUND_TEMPLATES).format(oid=oid)}


def invoice_case(rng: random.Random, systems: MockSystems) -> dict[str, Any]:
    return {"invoice_id": rng.choice(sorted(systems.invoices))}


def hr_case(rng: random.Random, systems: MockSystems) -> dict[str, Any]:
    return {"employee_id": rng.choice(sorted(systems.employees)), "question": _pick(rng, HR_QUESTIONS)}


GENERATORS: dict[str, Callable[[random.Random, MockSystems], dict[str, Any]]] = {
    "customer_refund": refund_case,
    "invoice_exception": invoice_case,
    "hr_onboarding": hr_case,
}


def generate_cases(process_id: str, n: int, systems: MockSystems, seed: int = 7) -> list[dict[str, Any]]:
    """Generate ``n`` simulated case inputs for one of the example processes."""
    if process_id not in GENERATORS:
        raise ValueError(f"no case generator for process '{process_id}'; available: {sorted(GENERATORS)}")
    rng = random.Random(seed)
    return [GENERATORS[process_id](rng, systems) for _ in range(n)]


def simulated_reviewer(refund_limit: float = 500.0, payment_limit: float = 15000.0) -> CallbackApprover:
    """A simulated human approver with signing limits: approves amounts up to the limit, denies above."""

    def decide(request: ApprovalRequest) -> bool:
        amount = float(request.args.get("amount", 0.0))
        limit = refund_limit if request.tool == "finance.issue_refund" else payment_limit
        return amount <= limit

    return CallbackApprover(decide, name="simulated reviewer")


def run_simulation(process, n: int = 200, seed: int = 7, approver=None, router=None) -> list:
    """Run ``n`` simulated cases through the real engine and return their traces.

    A fresh set of mock systems is created from ``seed`` so that runs are reproducible.
    """
    from .engine import WorkflowEngine
    from .tools import build_default_registry

    registry, systems = build_default_registry(seed=11)
    engine = WorkflowEngine(process, registry, approver=approver or simulated_reviewer(), router=router)
    cases = generate_cases(process.id, n, systems, seed)
    return [engine.run(inputs, case_id=f"{process.id}-{i + 1:04d}").trace for i, inputs in enumerate(cases)]
