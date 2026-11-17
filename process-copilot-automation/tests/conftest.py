from pathlib import Path

import pytest

from process_copilot.engine import WorkflowEngine
from process_copilot.process import load_process
from process_copilot.simulation import simulated_reviewer
from process_copilot.tools import build_default_registry

ROOT = Path(__file__).resolve().parents[1]
PROCESSES = ROOT / "examples" / "processes"


@pytest.fixture()
def systems_registry():
    registry, systems = build_default_registry()
    return registry, systems


@pytest.fixture()
def make_engine(systems_registry):
    registry, _ = systems_registry

    def _make(name: str, approver="reviewer", **kwargs) -> WorkflowEngine:
        process = load_process(PROCESSES / f"{name}.yaml")
        for key in ("cost_budget_usd",):
            if key in kwargs:
                setattr(process, key, kwargs.pop(key))
        chosen = simulated_reviewer() if approver == "reviewer" else approver
        return WorkflowEngine(process, registry, approver=chosen, **kwargs)

    return _make


def find_order(systems, predicate):
    return next(o for o in systems.orders.values() if predicate(o))


def find_invoice(systems, predicate):
    return next(i for i in systems.invoices.values() if predicate(i))
