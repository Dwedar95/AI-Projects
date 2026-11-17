"""Well-Architected style checklist engine.

An architecture is described in YAML. Rules are data (``data/waf_rules.yaml``):
each rule has a pillar, a severity and a condition written in a small
declarative language. The engine evaluates every rule, scores each pillar by
severity-weighted pass rate and applies a gate.

Condition language
------------------
Leaf:        ``{path: network.private_endpoints, op: eq, value: true}``
Combinators: ``{all: [...]}``, ``{any: [...]}``, ``{not: {...}}``
Quantifiers: ``{path: models, every: {...}}`` and ``{path: models, some: {...}}``
             evaluate the inner condition on each list item.

A leaf whose path is absent evaluates to *unknown*. Unknown counts as a failure
in the score (undocumented controls are not credited) but is reported
separately so the reader can tell "not done" from "not described".
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

PILLARS = ("security", "reliability", "cost", "operations", "performance", "responsible_ai")
SEVERITY_WEIGHT = {"critical": 5, "high": 3, "medium": 2, "low": 1}
_MISSING = object()


class Status(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class RuleError(ValueError):
    """Raised for malformed rules or conditions."""


def lookup(data: Any, path: str) -> Any:
    """Resolve a dotted path. Returns the ``_MISSING`` sentinel when absent."""
    current = data
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return _MISSING
    return current


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _compare(op: Callable[[float, float], bool]) -> Callable[[Any, Any], bool]:
    def run(actual: Any, expected: Any) -> bool:
        return _is_number(actual) and _is_number(expected) and op(actual, expected)

    return run


def _contains(actual: Any, expected: Any) -> bool:
    return isinstance(actual, list | str | dict) and expected in actual


OPERATORS: dict[str, Callable[[Any, Any], bool]] = {
    "eq": lambda actual, expected: actual == expected and isinstance(actual, bool) == isinstance(expected, bool),
    "ne": lambda actual, expected: actual != expected,
    "in": lambda actual, expected: actual in expected,
    "not_in": lambda actual, expected: actual not in expected,
    "gte": _compare(lambda a, b: a >= b),
    "lte": _compare(lambda a, b: a <= b),
    "gt": _compare(lambda a, b: a > b),
    "lt": _compare(lambda a, b: a < b),
    "truthy": lambda actual, _expected: bool(actual),
    "falsy": lambda actual, _expected: not actual,
    "contains": _contains,
    "contains_all": lambda actual, expected: isinstance(actual, list) and all(item in actual for item in expected),
    "len_gte": lambda actual, expected: hasattr(actual, "__len__") and len(actual) >= expected,
}


def evaluate(condition: dict[str, Any], data: Any) -> bool | None:
    """Evaluate a condition with three-valued logic: True, False or None (unknown)."""
    if not isinstance(condition, dict):
        raise RuleError(f"condition must be a mapping, got {condition!r}")
    if "all" in condition:
        results = [evaluate(c, data) for c in condition["all"]]
        if any(r is False for r in results):
            return False
        return None if any(r is None for r in results) else True
    if "any" in condition:
        results = [evaluate(c, data) for c in condition["any"]]
        if any(r is True for r in results):
            return True
        return None if any(r is None for r in results) else False
    if "not" in condition:
        result = evaluate(condition["not"], data)
        return None if result is None else not result
    if "path" not in condition:
        raise RuleError(f"condition needs 'path', 'all', 'any' or 'not': {condition!r}")

    value = lookup(data, condition["path"])
    if "exists" in condition:
        return (value is not _MISSING) == bool(condition["exists"])
    if value is _MISSING:
        return None
    for quantifier in ("every", "some"):
        if quantifier in condition:
            if not isinstance(value, list):
                return False
            results = [evaluate(condition[quantifier], item) for item in value]
            if quantifier == "every":
                if not results or any(r is False for r in results):
                    return False
                return None if any(r is None for r in results) else True
            if any(r is True for r in results):
                return True
            return None if any(r is None for r in results) else False
    op = condition.get("op")
    if op not in OPERATORS:
        raise RuleError(f"unknown operator {op!r} in {condition!r}")
    return OPERATORS[op](value, condition.get("value"))


@dataclass(frozen=True)
class Rule:
    id: str
    pillar: str
    severity: str
    title: str
    check: dict[str, Any]
    recommendation: str = ""
    applies_when: dict[str, Any] | None = None

    @property
    def weight(self) -> int:
        return SEVERITY_WEIGHT[self.severity]


@dataclass(frozen=True)
class RuleResult:
    rule: Rule
    status: Status


@dataclass(frozen=True)
class PillarScore:
    pillar: str
    passed: int
    applicable: int
    earned_weight: int
    total_weight: int

    @property
    def score(self) -> float:
        """Severity-weighted pass rate in percent. 100 when nothing applies."""
        return 100.0 * self.earned_weight / self.total_weight if self.total_weight else 100.0


@dataclass
class Assessment:
    name: str
    results: list[RuleResult]
    threshold: float = 80.0
    pillar_threshold: float = 60.0
    pillars: list[PillarScore] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.pillars = [self._pillar_score(p) for p in PILLARS if any(r.rule.pillar == p for r in self.results)]

    def _pillar_score(self, pillar: str) -> PillarScore:
        applicable = [r for r in self.results if r.rule.pillar == pillar and r.status != Status.NOT_APPLICABLE]
        passed = [r for r in applicable if r.status == Status.PASS]
        return PillarScore(
            pillar,
            len(passed),
            len(applicable),
            sum(r.rule.weight for r in passed),
            sum(r.rule.weight for r in applicable),
        )

    @property
    def overall(self) -> float:
        total = sum(p.total_weight for p in self.pillars)
        return 100.0 * sum(p.earned_weight for p in self.pillars) / total if total else 100.0

    @property
    def gaps(self) -> list[RuleResult]:
        """Failed and unknown rules, most severe first."""
        open_items = [r for r in self.results if r.status in (Status.FAIL, Status.UNKNOWN)]
        return sorted(open_items, key=lambda r: (-r.rule.weight, r.rule.id))

    @property
    def critical_gaps(self) -> list[RuleResult]:
        return [r for r in self.gaps if r.rule.severity == "critical"]

    @property
    def passed(self) -> bool:
        """Gate: overall score, every pillar above its floor and no critical gap."""
        return (
            self.overall >= self.threshold
            and all(p.score >= self.pillar_threshold for p in self.pillars)
            and not self.critical_gaps
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "overall": round(self.overall, 1),
            "threshold": self.threshold,
            "pillar_threshold": self.pillar_threshold,
            "pillars": [asdict(p) | {"score": round(p.score, 1)} for p in self.pillars],
            "results": [
                {
                    "id": r.rule.id,
                    "pillar": r.rule.pillar,
                    "severity": r.rule.severity,
                    "title": r.rule.title,
                    "status": r.status.value,
                }
                for r in self.results
            ],
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Architecture assessment: {self.name}",
            "",
            f"- Gate: {'PASS' if self.passed else 'FAIL'} "
            f"(overall >= {self.threshold:.0f}, every pillar >= {self.pillar_threshold:.0f}, no critical gap)",
            f"- Overall score: {self.overall:.1f} / 100",
            f"- Rules evaluated: {len(self.results)}, open gaps: {len(self.gaps)}, "
            f"critical gaps: {len(self.critical_gaps)}",
            "",
            "| Pillar | Score | Passed | Applicable |",
            "| --- | ---: | ---: | ---: |",
        ]
        for p in self.pillars:
            lines.append(f"| {p.pillar} | {p.score:.1f} | {p.passed} | {p.applicable} |")
        lines += ["", "## Gaps", ""]
        if not self.gaps:
            lines.append("No gaps.")
        else:
            lines += [
                "| Rule | Pillar | Severity | Status | Finding | Recommendation |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
            for r in self.gaps:
                status = "not documented" if r.status == Status.UNKNOWN else "not met"
                lines.append(
                    f"| {r.rule.id} | {r.rule.pillar} | {r.rule.severity} | {status} "
                    f"| {r.rule.title} | {r.rule.recommendation} |"
                )
        return "\n".join(lines) + "\n"


def parse_rules(raw: list[dict[str, Any]]) -> list[Rule]:
    """Validate raw rule mappings and build :class:`Rule` objects."""
    rules: list[Rule] = []
    seen: set[str] = set()
    for item in raw:
        missing = {"id", "pillar", "severity", "title", "check"} - set(item)
        if missing:
            raise RuleError(f"rule {item.get('id', '?')} is missing {sorted(missing)}")
        if item["id"] in seen:
            raise RuleError(f"duplicate rule id {item['id']}")
        if item["pillar"] not in PILLARS:
            raise RuleError(f"rule {item['id']}: unknown pillar {item['pillar']!r}")
        if item["severity"] not in SEVERITY_WEIGHT:
            raise RuleError(f"rule {item['id']}: unknown severity {item['severity']!r}")
        seen.add(item["id"])
        rules.append(
            Rule(
                id=item["id"],
                pillar=item["pillar"],
                severity=item["severity"],
                title=item["title"],
                check=item["check"],
                recommendation=item.get("recommendation", ""),
                applies_when=item.get("applies_when"),
            )
        )
    return rules


def load_rules(path: str | Path | None = None) -> list[Rule]:
    """Load rules from a YAML file. Without a path the bundled rule set is used."""
    if path is None:
        text = resources.files("archkit.data").joinpath("waf_rules.yaml").read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    return parse_rules(yaml.safe_load(text)["rules"])


def assess(
    architecture: dict[str, Any],
    rules: list[Rule] | None = None,
    threshold: float = 80.0,
    pillar_threshold: float = 60.0,
) -> Assessment:
    """Evaluate every rule against an architecture description."""
    results: list[RuleResult] = []
    for rule in rules if rules is not None else load_rules():
        if rule.applies_when is not None and evaluate(rule.applies_when, architecture) is not True:
            results.append(RuleResult(rule, Status.NOT_APPLICABLE))
            continue
        outcome = evaluate(rule.check, architecture)
        status = Status.UNKNOWN if outcome is None else (Status.PASS if outcome else Status.FAIL)
        results.append(RuleResult(rule, status))
    return Assessment(str(architecture.get("name", "unnamed")), results, threshold, pillar_threshold)


def load_architecture(path: str | Path) -> dict[str, Any]:
    """Read an architecture description YAML file."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuleError("architecture description must be a YAML mapping")
    return data
