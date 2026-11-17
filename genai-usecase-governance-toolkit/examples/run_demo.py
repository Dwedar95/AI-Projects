"""End-to-end demo on the fictional example portfolio. Prints real CLI output.

Usage: python examples/run_demo.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from genai_governance.cli import main  # noqa: E402

os.chdir(HERE.parent)
PORTFOLIO = "examples/portfolio.yaml"
PREDICTIONS = "examples/data/predictions_synthetic.csv"


def run(title: str, argv: list[str]) -> None:
    print(f"\n=== {title} ===\n$ genai-gov {' '.join(argv)}\n")
    main(argv)


if __name__ == "__main__":
    run("1. Prioritisation", ["prioritise", PORTFOLIO])
    run("2. Sensitivity of the ranking", ["sensitivity", PORTFOLIO])
    run("3. Build / buy / partner", ["build-buy", PORTFOLIO, "--use-case", "UC-04"])
    run("4. Business case", ["roi", PORTFOLIO, "--use-case", "UC-01"])
    run("5. Risk triage (portfolio)", ["triage", PORTFOLIO])
    run("6. Risk triage (one use case)", ["triage", PORTFOLIO, "--use-case", "UC-03"])
    run("7. Responsible AI evaluation (synthetic predictions)", ["fairness", PREDICTIONS])
    with tempfile.TemporaryDirectory() as tmp:
        print("\n=== 8. Reports ===\n$ genai-gov report examples/portfolio.yaml --out <tmpdir>\n")
        main(["report", PORTFOLIO, "--out", tmp])
        print("\n--- roadmap.md ---\n")
        print((Path(tmp) / "roadmap.md").read_text(encoding="utf-8"))
