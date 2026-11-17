"""Regenerate the committed synthetic benchmark files under ``benchmarks/``.

The generator is deterministic, so running this script reproduces the committed
JSONL files byte for byte (``tests/test_datasets.py`` checks that).
"""

from __future__ import annotations

from pathlib import Path

from llmops_gate.datasets import generate_tickets, write_jsonl

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"


def main() -> None:
    write_jsonl(
        BENCHMARKS / "intent_benchmark.jsonl", generate_tickets(400, seed=42, id_prefix="bench")
    )
    print("wrote benchmarks/intent_benchmark.jsonl (400 synthetic tickets, seed 42)")


if __name__ == "__main__":
    main()
