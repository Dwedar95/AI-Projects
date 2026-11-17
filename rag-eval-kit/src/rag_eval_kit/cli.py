"""Command line interface: ``rag-eval-kit eval | query | serve``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from rag_eval_kit.documents import bundled_corpus_dir, load_corpus
from rag_eval_kit.evaluation import apply_gate, load_qa, load_thresholds, render_markdown, run_evaluation
from rag_eval_kit.pipeline import PipelineConfig, RagPipeline


def _config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(chunker=args.chunker, embedder=args.embedder, mode=args.mode, rerank=not args.no_rerank)


def _documents(args: argparse.Namespace):
    return load_corpus(args.corpus) if args.corpus else load_corpus(bundled_corpus_dir())


def _cmd_eval(args: argparse.Namespace) -> int:
    report = run_evaluation(
        documents=load_corpus(args.corpus) if args.corpus else None,
        examples=load_qa(args.qa) if args.qa else None,
        config=_config(args),
        latency_repeats=args.latency_repeats,
    )
    gate = apply_gate(report, load_thresholds(args.gate))
    markdown = render_markdown(report, gate)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.md").write_text(markdown, encoding="utf-8")
        payload = report.to_dict() | {"gate": gate.to_dict()}
        (out / "report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(markdown)
    return 0 if gate.passed else 1


def _cmd_query(args: argparse.Namespace) -> int:
    pipeline = RagPipeline(_config(args))
    pipeline.ingest(_documents(args))
    response = pipeline.answer(args.question)
    print(response.answer.text)
    for citation in response.answer.citations:
        print(f"  [{citation.marker}] {citation.doc_id} / {citation.section} ({citation.chunk_id})")
    if response.flags:
        print("  flags: " + ", ".join(response.flags))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from rag_eval_kit.audit import AuditLog
    from rag_eval_kit.service import create_app, keys_from_env

    pipeline = RagPipeline(_config(args))
    pipeline.ingest(_documents(args))
    audit = AuditLog(os.environ.get("RAG_AUDIT_LOG", "audit/audit.jsonl"))
    uvicorn.run(create_app(pipeline, keys_from_env(), audit), host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-eval-kit", description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--corpus", help="directory with .md or .txt documents (default: bundled synthetic corpus)")
    common.add_argument("--chunker", choices=["fixed", "sentence", "heading"], default="heading")
    common.add_argument("--embedder", choices=["hashing", "tfidf-svd", "azure-openai"], default="hashing")
    common.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    common.add_argument("--no-rerank", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)

    evaluate = commands.add_parser("eval", parents=[common], help="run the evaluation and the gate")
    evaluate.add_argument("--qa", help="QA set in JSON Lines (default: bundled synthetic set)")
    evaluate.add_argument("--gate", help="gate thresholds YAML (default: bundled example gate)")
    evaluate.add_argument("--out", help="directory for report.md and report.json")
    evaluate.add_argument("--latency-repeats", type=int, default=5)
    evaluate.set_defaults(handler=_cmd_eval)

    query = commands.add_parser("query", parents=[common], help="answer one question on the command line")
    query.add_argument("question")
    query.set_defaults(handler=_cmd_query)

    serve = commands.add_parser("serve", parents=[common], help="start the HTTP service (needs RAG_API_KEYS)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(handler=_cmd_serve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns the process exit code."""
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as error:
        sys.stderr.write(f"rag-eval-kit: error: {error}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
