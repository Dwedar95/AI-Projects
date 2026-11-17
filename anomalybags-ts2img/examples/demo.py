"""Runnable demo: fit AnomalyBags on a synthetic benchmark and print real metrics.

Run with ``python examples/demo.py``. All data is synthetic.
"""

from anomalybags.cli import run_demo

if __name__ == "__main__":
    run_demo(seed=0, difficulty=0.6, n_estimators=16)
