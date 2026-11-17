"""Release gate and report rendering."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rag_eval_kit.evaluation.harness import DATA_DIR, EvalReport

_OPS = {">=": operator.ge, "<=": operator.le, ">": operator.gt, "<": operator.lt, "==": operator.eq}


@dataclass(frozen=True)
class GateCheck:
    metric: str
    op: str
    threshold: float
    actual: float | None

    @property
    def passed(self) -> bool:
        """A missing metric fails the check."""
        return self.actual is not None and _OPS[self.op](self.actual, self.threshold)


@dataclass(frozen=True)
class GateResult:
    checks: tuple[GateCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failed(self) -> list[GateCheck]:
        return [check for check in self.checks if not check.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [
                {"metric": c.metric, "op": c.op, "threshold": c.threshold, "actual": c.actual, "passed": c.passed}
                for c in self.checks
            ],
        }


def load_thresholds(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load gate checks from YAML. Without a path the bundled example gate is used."""
    source = Path(path) if path else DATA_DIR / "eval_gate.yaml"
    checks = yaml.safe_load(source.read_text(encoding="utf-8"))["checks"]
    for check in checks:
        if check.get("op") not in _OPS:
            raise ValueError(f"unknown gate operator {check.get('op')!r} for metric {check.get('metric')!r}")
    return checks


def apply_gate(report: EvalReport, thresholds: list[dict[str, Any]] | None = None) -> GateResult:
    """Compare report metrics with thresholds."""
    checks = thresholds if thresholds is not None else load_thresholds()
    return GateResult(
        tuple(GateCheck(c["metric"], c["op"], float(c["threshold"]), report.metric(c["metric"])) for c in checks)
    )


def _table(header: list[str], rows: list[list[Any]]) -> list[str]:
    def cell(value: Any) -> str:
        return f"{value:.3f}" if isinstance(value, float) else str(value)

    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] + ["---:"] * (len(header) - 1)) + " |"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return lines


def render_markdown(report: EvalReport, gate: GateResult | None = None) -> str:
    """Render the evaluation report as markdown."""
    meta = report.meta
    cfg = meta["config"]
    lines = [
        "# RAG evaluation report",
        "",
        f"- Data: {meta['data']} ({meta['documents']} documents, {meta['chunks']} chunks, "
        f"{meta['qa_examples']} questions: {meta['qa_by_kind']})",
        f"- Pipeline: chunker={cfg['chunker']}, embedder={cfg['embedder']}, mode={cfg['mode']}, "
        f"rerank={cfg['rerank']}, top_k={cfg['top_k']}, generator={meta['generator']}",
        f"- rag-eval-kit {meta['version']}, Python {meta['python']}",
    ]
    if gate is not None:
        passed_checks = len(gate.checks) - len(gate.failed)
        lines.append(f"- Gate: {'PASS' if gate.passed else 'FAIL'} ({passed_checks}/{len(gate.checks)} checks)")

    metric_names = ["recall@1", "recall@3", "recall@5", "mrr@10", "ndcg@10"]
    lines += ["", "## Retrieval: BM25 vs dense vs hybrid", ""]
    lines += _table(
        ["Configuration"] + metric_names,
        [[name] + [float(scores[k]) for k in metric_names] for name, scores in report.retrieval.items()],
    )
    configs = list(report.retrieval)
    lines += ["", "Recall@5 by question kind:", ""]
    lines += _table(
        ["Kind", "n"] + configs,
        [[kind, int(row["n"])] + [float(row[c]) for c in configs] for kind, row in report.retrieval_by_kind.items()],
    )
    columns = list(next(iter(report.embedders.values())))
    lines += ["", "## Offline embedders (BM25 is the same in both rows)", ""]
    lines += _table(
        ["Embedder"] + columns,
        [[name] + [float(row[c]) for c in columns] for name, row in report.embedders.items()],
    )
    lines += ["", f"## Chunking strategies (mode={cfg['mode']}, rerank={cfg['rerank']})", ""]
    lines += _table(
        ["Strategy", "chunks", "recall@5", "mrr@10"],
        [
            [name, int(row["chunks"]), float(row["recall@5"]), float(row["mrr@10"])]
            for name, row in report.chunking.items()
        ],
    )
    for title, section in (("Answers", report.answers), ("Safety", report.safety), ("Latency", report.latency)):
        lines += ["", f"## {title}", ""]
        lines += _table(["Metric", "Value"], [[name, value] for name, value in section.items()])
    if gate is not None:
        lines += ["", "## Gate", "", "| Metric | Rule | Actual | Result |", "| --- | --- | ---: | --- |"]
        for check in gate.checks:
            actual = "missing" if check.actual is None else f"{check.actual:.3f}"
            lines.append(
                f"| {check.metric} | {check.op} {check.threshold:g} | {actual} | {'pass' if check.passed else 'FAIL'} |"
            )
    if report.failures:
        counts: dict[str, int] = {}
        for failure in report.failures:
            counts[failure["issue"]] = counts.get(failure["issue"], 0) + 1
        lines += ["", "## Answer failures", ""]
        lines += [f"- {issue}: {count}" for issue, count in sorted(counts.items())]
    return "\n".join(lines) + "\n"
