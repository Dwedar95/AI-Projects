"""Tool registry and mock internal systems (CRM, finance, ticketing, HR).

The systems are in-process Python objects with in-memory state and FICTIONAL
seed data generated from a fixed seed. They stand in for internal APIs so that
action steps, approvals and side effects can be demonstrated and tested.
"""

from __future__ import annotations

import inspect
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


class ToolError(RuntimeError):
    """Raised when a tool is unknown, called with wrong arguments or fails."""


@dataclass(frozen=True)
class Tool:
    """A registered callable with metadata used by the policy engine and the trace."""

    name: str
    fn: Callable[..., dict[str, Any]]
    system: str
    side_effect: bool
    description: str = ""


class ToolRegistry:
    """Name to callable mapping. Tools are called with keyword arguments and return a dict."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, name: str, fn: Callable[..., dict[str, Any]], system: str, side_effect: bool,
                 description: str = "") -> None:
        if name in self._tools:
            raise ValueError(f"tool already registered: {name}")
        self._tools[name] = Tool(name, fn, system, side_effect, description or (inspect.getdoc(fn) or ""))

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ToolError(f"unknown tool: {name}")
        return self._tools[name]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Call a tool. Argument mismatches are reported as :class:`ToolError`."""
        tool = self.get(name)
        try:
            inspect.signature(tool.fn).bind(**args)
        except TypeError as exc:
            raise ToolError(f"bad arguments for {name}: {exc}") from exc
        return tool.fn(**args)


@dataclass
class MockSystems:
    """In-memory state of the mock systems, created from a seed. All records are fictional."""

    seed: int = 11
    customers: dict[str, dict[str, Any]] = field(default_factory=dict)
    orders: dict[str, dict[str, Any]] = field(default_factory=dict)
    invoices: dict[str, dict[str, Any]] = field(default_factory=dict)
    purchase_orders: dict[str, dict[str, Any]] = field(default_factory=dict)
    employees: dict[str, dict[str, Any]] = field(default_factory=dict)
    tickets: list[dict[str, Any]] = field(default_factory=list)
    refunds: list[dict[str, Any]] = field(default_factory=list)
    payments: list[dict[str, Any]] = field(default_factory=list)
    onboarding_tasks: list[dict[str, Any]] = field(default_factory=list)
    interactions: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        rng = random.Random(self.seed)
        for i in range(20):
            cid = f"C-{1001 + i}"
            tier = rng.choice(["standard", "gold"])
            self.customers[cid] = {"customer_id": cid, "name": f"Customer {1001 + i}", "tier": tier}
        for i in range(60):
            oid = f"ORD-{1000 + i}"
            self.orders[oid] = {
                "order_id": oid, "found": True,
                "customer_id": f"C-{1001 + rng.randrange(20)}",
                "amount": round(rng.choice([rng.uniform(15, 100), rng.uniform(100, 400), rng.uniform(400, 1500)]), 2),
                "days_since_delivery": rng.randrange(0, 46),
                "status": "delivered" if rng.random() < 0.88 else "in_transit",
            }
        notes = {
            "match": "Invoice matches the purchase order.",
            "small": "Rounding difference on freight.",
            "price": "Unit price on the invoice is higher than the agreed contract price.",
            "quantity": "Quantity invoiced exceeds the quantity on the goods receipt.",
            "missing": "No purchase order reference on the invoice.",
        }
        kinds = ["match"] * 11 + ["small"] * 3 + ["price"] * 3 + ["quantity"] * 2 + ["missing"]
        for i in range(40):
            kind = rng.choice(kinds)
            po_amount = round(rng.choice([rng.uniform(200, 4000), rng.uniform(4000, 20000)]), 2)
            factor = {"match": 1.0, "small": 1 + rng.uniform(0.001, 0.015), "price": 1 + rng.uniform(0.05, 0.2),
                      "quantity": 1 + rng.uniform(0.1, 0.3), "missing": 1.0}[kind]
            inv, po, supplier = f"INV-{2000 + i}", f"PO-{5000 + i}", f"Supplier {i % 7 + 1}"
            if kind != "missing":
                self.purchase_orders[po] = {"po_id": po, "found": True, "amount": po_amount, "supplier": supplier}
            self.invoices[inv] = {"invoice_id": inv, "found": True, "po_id": po if kind != "missing" else "",
                                  "amount": round(po_amount * factor, 2), "note": notes[kind], "supplier": supplier}
        for i in range(20):
            eid = f"E-{3000 + i}"
            self.employees[eid] = {"employee_id": eid, "found": True, "name": f"Employee {3000 + i}",
                                   "department": rng.choice(["Sales", "Engineering", "Finance", "Operations"]),
                                   "start_date": f"2026-{rng.randrange(1, 13):02d}-01"}

    # CRM
    def get_customer(self, customer_id: str) -> dict[str, Any]:
        """Read a customer record."""
        return self.customers.get(customer_id, {"customer_id": customer_id, "found": False})

    def get_order(self, order_id: str) -> dict[str, Any]:
        """Read an order with amount, delivery age in days and status."""
        return self.orders.get(order_id, {"order_id": order_id, "found": False, "amount": 0.0,
                                          "days_since_delivery": -1, "status": "unknown"})

    def log_interaction(self, customer_id: str, summary: str) -> dict[str, Any]:
        """Append an interaction note to the customer history."""
        self.interactions.append({"customer_id": customer_id, "summary": summary})
        return {"logged": True, "count": len(self.interactions)}

    # Finance
    def issue_refund(self, order_id: str, amount: float) -> dict[str, Any]:
        """Refund an order. Fails for unknown orders, non-positive amounts or amounts above the order value."""
        order = self.orders.get(order_id)
        if order is None:
            raise ToolError(f"cannot refund unknown order {order_id}")
        if amount <= 0 or amount > order["amount"] + 1e-9:
            raise ToolError(f"refund amount {amount} is not valid for order {order_id}")
        refund = {"refund_id": f"RF-{len(self.refunds) + 1:04d}", "order_id": order_id, "amount": amount, "status": "issued"}
        self.refunds.append(refund)
        return refund

    def get_invoice(self, invoice_id: str) -> dict[str, Any]:
        """Read an invoice."""
        return self.invoices.get(invoice_id, {"invoice_id": invoice_id, "found": False, "po_id": "", "amount": 0.0,
                                              "note": "", "supplier": ""})

    def get_purchase_order(self, po_id: str) -> dict[str, Any]:
        """Read a purchase order. ``found`` is false when the id is empty or unknown."""
        return self.purchase_orders.get(po_id, {"po_id": po_id, "found": False, "amount": 0.0, "supplier": ""})

    def approve_payment(self, invoice_id: str, amount: float) -> dict[str, Any]:
        """Release an invoice for payment."""
        if invoice_id not in self.invoices:
            raise ToolError(f"cannot pay unknown invoice {invoice_id}")
        payment = {"payment_id": f"PAY-{len(self.payments) + 1:04d}", "invoice_id": invoice_id, "amount": amount,
                   "status": "released"}
        self.payments.append(payment)
        return payment

    # Ticketing
    def create_ticket(self, queue: str, subject: str, body: str = "", status: str = "open") -> dict[str, Any]:
        """Create a ticket in a queue."""
        ticket = {"ticket_id": f"T-{len(self.tickets) + 1:05d}", "queue": queue, "subject": subject, "body": body,
                  "status": status}
        self.tickets.append(ticket)
        return ticket

    # HR
    def get_employee(self, employee_id: str) -> dict[str, Any]:
        """Read an employee record."""
        return self.employees.get(employee_id, {"employee_id": employee_id, "found": False, "name": "",
                                                "department": "", "start_date": ""})

    def create_onboarding_task(self, employee_id: str, task: str) -> dict[str, Any]:
        """Create an onboarding task for an employee."""
        if employee_id not in self.employees:
            raise ToolError(f"unknown employee {employee_id}")
        item = {"task_id": f"OB-{len(self.onboarding_tasks) + 1:04d}", "employee_id": employee_id, "task": task}
        self.onboarding_tasks.append(item)
        return item


def build_default_registry(seed: int = 11) -> tuple[ToolRegistry, MockSystems]:
    """Create the mock systems and register their operations as tools."""
    systems = MockSystems(seed=seed)
    registry = ToolRegistry()
    registry.register("crm.get_customer", systems.get_customer, "CRM", side_effect=False)
    registry.register("crm.get_order", systems.get_order, "CRM", side_effect=False)
    registry.register("crm.log_interaction", systems.log_interaction, "CRM", side_effect=True)
    registry.register("finance.issue_refund", systems.issue_refund, "Finance", side_effect=True)
    registry.register("finance.get_invoice", systems.get_invoice, "Finance", side_effect=False)
    registry.register("finance.get_purchase_order", systems.get_purchase_order, "Finance", side_effect=False)
    registry.register("finance.approve_payment", systems.approve_payment, "Finance", side_effect=True)
    registry.register("ticketing.create_ticket", systems.create_ticket, "Ticketing", side_effect=True)
    registry.register("hr.get_employee", systems.get_employee, "HR", side_effect=False)
    registry.register("hr.create_onboarding_task", systems.create_onboarding_task, "HR", side_effect=True)
    return registry, systems
