"""Runnable demo: BEM check plus a reduced-budget surrogate workflow.

Run with ``python examples/demo.py``. The full workflow that produces the numbers in
the README is ``python -m wtsurrogate.cli pipeline``.
"""

from wtsurrogate.cli import run_demo

if __name__ == "__main__":
    run_demo()
