"""Generate the SYNTHETIC prediction file used by the fairness demo.

The data imitates the output of a CV screening classifier. It is produced from
random numbers with a fixed seed and contains no real persons. Group B gets a
lower score on purpose so that the fairness metrics have something to detect.

Usage: python examples/generate_predictions.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "data" / "predictions_synthetic.csv"


def generate(n: int = 2000, seed: int = 7) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    group = rng.choice(["A", "B"], size=n, p=[0.6, 0.4])
    qualified = rng.random(n) < 0.35  # same base rate in both groups
    signal = np.where(qualified, 0.9, -0.9) + rng.normal(0, 1.0, n)
    signal = signal - np.where(group == "B", 0.45, 0.0)  # injected bias against group B
    score = 1.0 / (1.0 + np.exp(-signal))
    pred = (score >= 0.5).astype(int)
    return [
        {"id": i, "group": group[i], "y_true": int(qualified[i]), "y_score": round(float(score[i]), 4),
         "y_pred": int(pred[i])}
        for i in range(n)
    ]


if __name__ == "__main__":
    rows = generate()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} synthetic rows to {OUT}")
