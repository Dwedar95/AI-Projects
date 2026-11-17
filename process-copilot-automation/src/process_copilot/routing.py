"""Model routing: cheap versus strong model by task complexity and cost budget, hosting by data class."""

from __future__ import annotations

from dataclasses import dataclass

from .process import Step
from .providers import Hosting, ModelSpec

# Illustrative prices and latencies for the mock catalog. They are NOT vendor price list values.
DEFAULT_CATALOG: tuple[ModelSpec, ...] = (
    ModelSpec("prop-small", "proprietary", "cheap", 0.00015, 0.0006, 150.0, 6.0),
    ModelSpec("prop-large", "proprietary", "strong", 0.0025, 0.0100, 400.0, 18.0),
    ModelSpec("oss-small", "open_source", "cheap", 0.00010, 0.0001, 200.0, 10.0),
    ModelSpec("oss-large", "open_source", "strong", 0.00090, 0.0009, 600.0, 25.0),
)

TASK_COMPLEXITY = {"classify": 1, "extract": 1, "summarise": 2, "generate": 3, "rag": 2}
DECLARED_COMPLEXITY = {"low": 1, "medium": 2, "high": 4}


class BudgetExceededError(RuntimeError):
    """Raised when even the cheap model does not fit into the remaining case budget."""


@dataclass(frozen=True)
class RoutingDecision:
    """Chosen model with the reason, for the trace."""

    model: ModelSpec
    complexity: int
    estimated_cost: float
    reason: str
    downgraded: bool = False


class ModelRouter:
    """Rule-based router.

    1. Hosting: steps with ``data_classification: confidential`` go to the
       open-source (self-hosted) models, everything else to ``default_hosting``.
    2. Tier: a complexity score is computed from the task type, the prompt
       length and an optional declared complexity. A score at or above
       ``strong_threshold`` selects the strong tier.
    3. Budget: if the strong model's estimated cost does not fit into the
       remaining case budget the step is downgraded to the cheap tier. If the
       cheap model does not fit either, :class:`BudgetExceededError` is raised.
    """

    def __init__(self, catalog: tuple[ModelSpec, ...] = DEFAULT_CATALOG, strong_threshold: int = 3,
                 default_hosting: Hosting = "proprietary"):
        self.catalog = catalog
        self.strong_threshold = strong_threshold
        self.default_hosting = default_hosting
        for hosting in ("proprietary", "open_source"):
            for tier in ("cheap", "strong"):
                self.model(hosting, tier)  # fail early on an incomplete catalog

    def model(self, hosting: str, tier: str) -> ModelSpec:
        """Return the catalog model for a hosting kind and tier."""
        for m in self.catalog:
            if m.hosting == hosting and m.tier == tier:
                return m
        raise ValueError(f"catalog has no {tier} model for hosting '{hosting}'")

    def complexity(self, step: Step, prompt_tokens: int) -> int:
        """Complexity score of a step: declared value if present, otherwise task base plus length bonus."""
        if step.complexity:
            return DECLARED_COMPLEXITY[step.complexity]
        base = TASK_COMPLEXITY["rag" if step.type == "rag" else (step.task or "generate")]
        return base + (2 if prompt_tokens > 1200 else 1 if prompt_tokens > 400 else 0)

    def route(self, step: Step, prompt_tokens: int, spent: float, budget: float) -> RoutingDecision:
        """Pick a model for the step given what the case has already spent."""
        hosting = "open_source" if step.data_classification == "confidential" else self.default_hosting
        score = self.complexity(step, prompt_tokens)
        remaining = budget - spent
        cheap = self.model(hosting, "cheap")
        cheap_cost = cheap.cost(prompt_tokens, step.max_output_tokens)
        why_hosting = "confidential data stays on self-hosted model" if hosting == "open_source" else f"{hosting} default"
        if score >= self.strong_threshold:
            strong = self.model(hosting, "strong")
            strong_cost = strong.cost(prompt_tokens, step.max_output_tokens)
            if strong_cost <= remaining:
                return RoutingDecision(strong, score, strong_cost,
                                       f"complexity {score} >= {self.strong_threshold}; {why_hosting}")
            if cheap_cost <= remaining:
                return RoutingDecision(cheap, score, cheap_cost,
                                       f"complexity {score} but budget left {remaining:.5f} < {strong_cost:.5f}; "
                                       f"downgraded; {why_hosting}", downgraded=True)
        elif cheap_cost <= remaining:
            return RoutingDecision(cheap, score, cheap_cost, f"complexity {score} < {self.strong_threshold}; {why_hosting}")
        raise BudgetExceededError(f"step {step.id}: remaining budget {remaining:.5f} does not cover {cheap_cost:.5f}")

    def escalate(self, decision: RoutingDecision, step: Step, prompt_tokens: int, spent: float,
                 budget: float) -> RoutingDecision | None:
        """Cascade from the cheap to the strong model of the same hosting, if the budget allows it."""
        if decision.model.tier == "strong":
            return None
        strong = self.model(decision.model.hosting, "strong")
        cost = strong.cost(prompt_tokens, step.max_output_tokens)
        if cost > budget - spent:
            return None
        return RoutingDecision(strong, decision.complexity, cost, "escalated after low confidence on cheap model")
