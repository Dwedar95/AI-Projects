"""Train the example intent classifier on synthetic tickets and export it to ONNX.

Usage: ``python examples/train_intent_model.py --out artifacts/candidate.onnx``
"""

from __future__ import annotations

import argparse

from llmops_gate.datasets import generate_tickets
from llmops_gate.training import save_model, train_intent_classifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--data-seed", type=int, default=1)
    parser.add_argument("--hidden", type=int, nargs="+", default=[64])
    args = parser.parse_args()

    records = generate_tickets(args.train_size, seed=args.data_seed, id_prefix="train")
    onnx_bytes, metadata = train_intent_classifier(records, hidden=tuple(args.hidden))
    path = save_model(args.out, onnx_bytes, metadata)
    print(
        f"wrote {path} ({len(onnx_bytes)} bytes), train_size={metadata['train_size']}, "
        f"train_accuracy={metadata['train_accuracy']}, epochs={metadata['epochs']}"
    )


if __name__ == "__main__":
    main()
