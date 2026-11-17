"""Command line interface: ``archkit estimate | assess | validate-bicep``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from archkit import __version__
from archkit.bicep import BicepParseError, validate_directory
from archkit.bicep.policy import DEFAULT_REQUIRED_TAGS, default_rules
from archkit.checklist import assess, load_architecture, load_rules
from archkit.estimate import build_estimate, load_estimate_input
from archkit.pricing import load_prices


def _emit(markdown: str, payload: dict, args: argparse.Namespace) -> None:
    """Print the report in the requested format and optionally write it to a file."""
    text = json.dumps(payload, indent=2) if args.format == "json" else markdown
    if args.output:
        Path(args.output).write_text(text + ("" if text.endswith("\n") else "\n"), encoding="utf-8")
    else:
        sys.stdout.write(text if text.endswith("\n") else text + "\n")


def _cmd_estimate(args: argparse.Namespace) -> int:
    estimate = build_estimate(load_estimate_input(args.workload), load_prices(args.prices))
    _emit(estimate.to_markdown(), estimate.to_dict(), args)
    return 0


def _cmd_assess(args: argparse.Namespace) -> int:
    assessment = assess(
        load_architecture(args.architecture),
        load_rules(args.rules),
        threshold=args.threshold,
        pillar_threshold=args.pillar_threshold,
    )
    _emit(assessment.to_markdown(), assessment.to_dict(), args)
    return 0 if assessment.passed else 1


def _cmd_validate(args: argparse.Namespace) -> int:
    required = [t for t in args.required_tags.split(",") if t]
    report = validate_directory(args.directory, default_rules(required))
    _emit(report.to_markdown(), report.to_dict(), args)
    if not report.passed:
        return 1
    return 1 if args.strict and report.warnings else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="archkit", description=__doc__)
    parser.add_argument("--version", action="version", version=f"archkit {__version__}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--format", choices=["markdown", "json"], default="markdown")
    common.add_argument("--output", help="write the report to this file instead of stdout")
    commands = parser.add_subparsers(dest="command", required=True)

    estimate = commands.add_parser("estimate", parents=[common], help="capacity and cost estimate")
    estimate.add_argument("workload", help="workload YAML file")
    estimate.add_argument("--prices", help="price book YAML (default: bundled illustrative placeholders)")
    estimate.set_defaults(handler=_cmd_estimate)

    assess_cmd = commands.add_parser("assess", parents=[common], help="score an architecture description")
    assess_cmd.add_argument("architecture", help="architecture description YAML file")
    assess_cmd.add_argument("--rules", help="rules YAML (default: bundled rule set)")
    assess_cmd.add_argument("--threshold", type=float, default=80.0, help="minimum overall score")
    assess_cmd.add_argument("--pillar-threshold", type=float, default=60.0, help="minimum score per pillar")
    assess_cmd.set_defaults(handler=_cmd_assess)

    validate = commands.add_parser("validate-bicep", parents=[common], help="static policy check of Bicep files")
    validate.add_argument("directory", help="directory that contains .bicep files")
    validate.add_argument("--required-tags", default=",".join(DEFAULT_REQUIRED_TAGS))
    validate.add_argument("--strict", action="store_true", help="treat warnings as failures")
    validate.set_defaults(handler=_cmd_validate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns the process exit code."""
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (FileNotFoundError, BicepParseError, KeyError, ValueError) as error:
        sys.stderr.write(f"archkit: error: {error}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
