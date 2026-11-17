"""Evaluation gates: threshold and regression checks that decide promotion."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class GateRule(BaseModel):
    """Checks on one metric.

    ``min`` and ``max`` are absolute limits. ``max_regression`` (absolute) and
    ``max_regression_pct`` (relative, in percent) compare against the baseline
    and are skipped when there is no baseline, which is the case for the first
    version of a model.
    """

    metric: str
    direction: Literal["higher_is_better", "lower_is_better"] = "higher_is_better"
    min: float | None = None
    max: float | None = None
    max_regression: float | None = Field(default=None, ge=0)
    max_regression_pct: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _has_a_check(self) -> GateRule:
        if all(
            v is None for v in (self.min, self.max, self.max_regression, self.max_regression_pct)
        ):
            raise ValueError(f"gate rule for '{self.metric}' defines no check")
        return self


class GateConfig(BaseModel):
    gates: list[GateRule] = Field(min_length=1)


class GateCheck(BaseModel):
    metric: str
    rule: str
    value: float | None
    limit: float | None
    passed: bool
    message: str


class GateDecision(BaseModel):
    passed: bool
    baseline_ref: str | None = None
    checks: list[GateCheck]

    def failures(self) -> list[str]:
        return [c.message for c in self.checks if not c.passed]


def load_gate_config(path: str | Path) -> GateConfig:
    """Load and validate a YAML gate configuration."""
    return GateConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def evaluate_gates(
    candidate: dict[str, float],
    baseline: dict[str, float] | None,
    config: GateConfig,
    baseline_ref: str | None = None,
) -> GateDecision:
    """Apply every rule to the candidate metrics and return the combined decision."""
    checks: list[GateCheck] = []

    def add(rule: GateRule, name: str, value: float, limit: float, ok: bool, text: str) -> None:
        checks.append(
            GateCheck(
                metric=rule.metric,
                rule=name,
                value=value,
                limit=round(limit, 6),
                passed=ok,
                message=f"{rule.metric} {text}",
            )
        )

    for rule in config.gates:
        value = candidate.get(rule.metric)
        if value is None:
            checks.append(
                GateCheck(
                    metric=rule.metric,
                    rule="present",
                    value=None,
                    limit=None,
                    passed=False,
                    message=f"{rule.metric} is missing from the candidate metrics",
                )
            )
            continue
        if rule.min is not None:
            add(rule, "min", value, rule.min, value >= rule.min, f"{value:g} >= min {rule.min:g}")
        if rule.max is not None:
            add(rule, "max", value, rule.max, value <= rule.max, f"{value:g} <= max {rule.max:g}")
        base = baseline.get(rule.metric) if baseline else None
        if base is None:
            continue
        higher = rule.direction == "higher_is_better"
        regression = (base - value) if higher else (value - base)
        if rule.max_regression is not None:
            limit = base - rule.max_regression if higher else base + rule.max_regression
            add(
                rule,
                "max_regression",
                value,
                limit,
                regression <= rule.max_regression + 1e-12,
                f"{value:g} vs baseline {base:g}: regression {max(regression, 0):g} "
                f"(allowed {rule.max_regression:g})",
            )
        if rule.max_regression_pct is not None:
            pct = 100.0 * regression / abs(base) if base else (0.0 if regression <= 0 else 1e9)
            factor = rule.max_regression_pct / 100.0
            limit = base * (1 - factor) if higher else base * (1 + factor)
            add(
                rule,
                "max_regression_pct",
                value,
                limit,
                pct <= rule.max_regression_pct + 1e-9,
                f"{value:g} vs baseline {base:g}: regression {max(pct, 0):.1f}% "
                f"(allowed {rule.max_regression_pct:g}%)",
            )
    return GateDecision(
        passed=all(c.passed for c in checks), baseline_ref=baseline_ref, checks=checks
    )


def decision_dict(decision: GateDecision) -> dict[str, Any]:
    """Plain-dict form stored on the registry entry."""
    return decision.model_dump()
