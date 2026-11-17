"""End-to-end demo: single cases with traces, then simulated adoption analytics.

All data is fictional, all models are the deterministic mock, all case logs are simulated.

Usage: python examples/run_demo.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
os.chdir(HERE.parent)

from process_copilot.cli import main  # noqa: E402

REFUND = "examples/processes/customer_refund.yaml"
INVOICE = "examples/processes/invoice_exception.yaml"
HR = "examples/processes/hr_onboarding.yaml"


def run(title: str, argv: list[str]) -> None:
    shown = " ".join(f'"{a}"' if " " in a else a for a in argv)
    print(f"\n=== {title} ===\n$ process-copilot {shown}\n")
    main(argv)


if __name__ == "__main__":
    run("1. Validate a process", ["validate", REFUND])
    run("2. Refund with cascade and approval", ["run", REFUND, "--input", "customer_id=C-1007", "--input",
                                               "message=Order ORD-1000 was faulty and I want it refunded."])
    run("3. Refund above the reviewer's limit", ["run", REFUND, "--input", "customer_id=C-1007", "--input",
                                                "message=Please refund order ORD-1001, it is not what I ordered."])
    run("4. Invoice exception", ["run", INVOICE, "--input", "invoice_id=INV-2000"])
    run("5. Sensitive HR question", ["run", HR, "--input", "employee_id=E-3001", "--input",
                                    "question=I need to report harassment by a colleague."])
    run("6. Simulated adoption analytics: refund", ["simulate", REFUND])
    run("7. Simulated adoption analytics: invoice", ["simulate", INVOICE])
    run("8. Simulated adoption analytics: HR", ["simulate", HR])
