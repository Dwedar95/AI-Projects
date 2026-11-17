"""Markdown report generation: roadmap with waves, risk register and RACI matrix."""

from __future__ import annotations

from dataclasses import dataclass

from .fairness import FairnessReport
from .models import Portfolio
from .risk_triage import DISCLAIMER, TriageResult, triage
from .scoring import ScoredUseCase, dimension_scores, score_portfolio


@dataclass
class WaveAssignment:
    """Placement of one use case on the roadmap."""

    use_case_id: str
    name: str
    wave: int
    rank: int
    tier: str
    reason: str


def triage_portfolio(portfolio: Portfolio) -> dict[str, TriageResult]:
    """Run the risk triage for every use case of the portfolio."""
    return {u.id: triage(u.description, u.triage, u.role) for u in portfolio.use_cases}


def plan_waves(
    portfolio: Portfolio,
    scored: list[ScoredUseCase] | None = None,
    tiers: dict[str, TriageResult] | None = None,
    capacity: int = 2,
    n_waves: int = 3,
) -> tuple[list[WaveAssignment], list[WaveAssignment]]:
    """Assign use cases to delivery waves. Returns (planned, excluded).

    Rules, applied in rank order:
    - prohibited use cases are excluded from the roadmap;
    - feasible use cases (quick wins and fill-ins) may start in wave 1;
    - strategic bets (high value, low feasibility) and high-risk use cases
      start in wave 2 at the earliest, to leave lead time for foundations and
      governance work, and deprioritised ones go to the last wave;
    - a use case is never scheduled before a use case it depends on;
    - each wave except the last holds at most ``capacity`` use cases.
    """
    if capacity < 1 or n_waves < 1:
        raise ValueError("capacity and n_waves must be at least 1")
    scored = scored or score_portfolio(portfolio)
    tiers = tiers or triage_portfolio(portfolio)
    by_id = {s.id: s for s in scored}
    load = {w: 0 for w in range(1, n_waves + 1)}
    placed: dict[str, WaveAssignment] = {}
    excluded: list[WaveAssignment] = []
    visiting: set[str] = set()

    def place(uid: str) -> int | None:
        if uid in placed:
            return placed[uid].wave
        if uid in visiting:
            raise ValueError(f"dependency cycle detected at use case {uid}")
        s, tier = by_id[uid], tiers[uid].tier
        if tier == "prohibited":
            if all(e.use_case_id != uid for e in excluded):
                excluded.append(WaveAssignment(uid, s.name, 0, s.rank, tier, "prohibited practice, not to be pursued"))
            return None
        visiting.add(uid)
        earliest, reason = 1, f"{s.quadrant}, rank {s.rank}"
        if tier == "high-risk":
            earliest, reason = 2, f"high-risk tier needs governance lead time, rank {s.rank}"
        elif s.quadrant == "deprioritise":
            earliest, reason = n_waves, "low value and low feasibility"
        elif s.quadrant == "strategic bet":
            earliest, reason = min(2, n_waves), f"strategic bet needs foundations first, rank {s.rank}"
        for dep in portfolio.get(uid).depends_on:
            dep_wave = place(dep)
            if dep_wave is None:
                visiting.discard(uid)
                excluded.append(WaveAssignment(uid, s.name, 0, s.rank, tier, f"depends on excluded use case {dep}"))
                return None
            if dep_wave > earliest:
                earliest, reason = dep_wave, f"waits for dependency {dep}"
        earliest = min(earliest, n_waves)
        wave = earliest
        while wave < n_waves and load[wave] >= capacity:
            wave += 1
        if wave != earliest:
            reason += f", moved from wave {earliest} by capacity"
        load[wave] += 1
        placed[uid] = WaveAssignment(uid, s.name, wave, s.rank, tier, reason)
        visiting.discard(uid)
        return wave

    for s in scored:
        place(s.id)
    planned = sorted(placed.values(), key=lambda a: (a.wave, a.rank))
    return planned, excluded


def roadmap_markdown(portfolio: Portfolio, planned: list[WaveAssignment], excluded: list[WaveAssignment]) -> str:
    """Render the roadmap as Markdown, one table per wave."""
    lines = [f"# Roadmap: {portfolio.name}", ""]
    if portfolio.fictional:
        lines += ["Note: this portfolio is fictional example data.", ""]
    for wave in sorted({a.wave for a in planned}):
        lines += [f"## Wave {wave}", "", "| Rank | Use case | Indicative tier | Placement reason |", "|---|---|---|---|"]
        lines += [f"| {a.rank} | {a.name} ({a.use_case_id}) | {a.tier} | {a.reason} |" for a in planned if a.wave == wave]
        lines.append("")
    if excluded:
        lines += ["## Not on the roadmap", ""]
        lines += [f"- {a.name} ({a.use_case_id}): {a.reason}" for a in excluded]
        lines.append("")
    lines += [f"> {DISCLAIMER}", ""]
    return "\n".join(lines)


@dataclass
class RiskEntry:
    """One row of the risk register. Severity is likelihood times impact (1..25)."""

    use_case_id: str
    risk: str
    likelihood: int
    impact: int
    mitigation: str
    owner: str

    @property
    def severity(self) -> int:
        return self.likelihood * self.impact

    @property
    def rating(self) -> str:
        return "high" if self.severity >= 15 else "medium" if self.severity >= 8 else "low"


MITIGATIONS: tuple[tuple[str, str], ...] = (
    ("data", "Data protection impact assessment, data minimisation, access controls and retention rules"),
    ("privacy", "Data protection impact assessment, data minimisation, access controls and retention rules"),
    ("regulat", "Legal review of the classification, obligations checklist tracked to closure before go-live"),
    ("reputation", "Human review of external output, content filters, staged rollout with monitoring"),
    ("halluc", "Grounding on approved sources, citation checks, evaluation set with acceptance thresholds"),
    ("bias", "Fairness evaluation per release, representative test data, documented thresholds"),
)


def _mitigation(criterion: str) -> str:
    for key, text in MITIGATIONS:
        if key in criterion.lower():
            return text
    return "Assign an owner, define a control and review it at each release gate"


def build_risk_register(
    portfolio: Portfolio,
    tiers: dict[str, TriageResult] | None = None,
    fairness: dict[str, FairnessReport] | None = None,
    min_score: float = 3.0,
) -> list[RiskEntry]:
    """Derive risk entries from risk criterion scores, the triage tier and fairness flags.

    Likelihood is the raw risk score. Impact grows with the value dimension,
    because a failure in a high-value use case costs more: 1 + round(4 * value).
    """
    tiers = tiers or triage_portfolio(portfolio)
    fairness = fairness or {}
    risk_criteria = [c for c in portfolio.scoring.criteria if c.dimension == "risk"]
    entries: list[RiskEntry] = []
    for uc in portfolio.use_cases:
        value = dimension_scores(uc, portfolio.scoring)["value"]
        impact = 1 + round(4 * value)
        owner = uc.owner or "Business owner"
        for c in risk_criteria:
            score = uc.scores[c.name]
            if score >= min_score:
                entries.append(RiskEntry(uc.id, f"Elevated {c.name.replace('_', ' ')}", round(score), impact,
                                         _mitigation(c.name), owner))
        tier = tiers[uc.id].tier
        if tier == "prohibited":
            entries.append(RiskEntry(uc.id, "Indicative triage: prohibited practice (Art. 5)", 5, 5,
                                     "Stop the use case and obtain legal advice", "Legal and compliance"))
        elif tier == "high-risk":
            entries.append(RiskEntry(uc.id, "Indicative triage: high-risk obligations not met at go-live", 3, 5,
                                     "Track the obligations checklist, assign human oversight, legal sign-off",
                                     "Legal and compliance"))
        elif tier == "limited":
            entries.append(RiskEntry(uc.id, "Transparency obligations (Art. 50) missed", 2, 3,
                                     "Add AI disclosure to the user interface and mark generated content", owner))
        report = fairness.get(uc.id)
        if report and report.flags:
            entries.append(RiskEntry(uc.id, "Fairness thresholds exceeded: " + "; ".join(report.flags), 4, 4,
                                     "Investigate root cause, rebalance or recalibrate, re-evaluate before release",
                                     "AI product lead"))
    entries.sort(key=lambda e: (-e.severity, e.use_case_id, e.risk))
    return entries


def risk_register_markdown(portfolio: Portfolio, entries: list[RiskEntry]) -> str:
    """Render the risk register as a Markdown table."""
    lines = [f"# Risk register: {portfolio.name}", ""]
    if portfolio.fictional:
        lines += ["Note: this portfolio is fictional example data.", ""]
    lines += ["Severity is likelihood times impact on a 1 to 5 scale each. High is 15 or more, medium is 8 to 14.", "",
              "| Use case | Risk | L | I | Severity | Rating | Mitigation | Owner |", "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {e.use_case_id} | {e.risk} | {e.likelihood} | {e.impact} | {e.severity} | {e.rating} | "
              f"{e.mitigation} | {e.owner} |" for e in entries]
    lines += ["", f"> {DISCLAIMER}", ""]
    return "\n".join(lines)


ROLES = ("Business owner", "AI product lead", "Engineering lead", "Data protection officer",
         "AI governance board", "Legal and compliance")

BASE_RACI: tuple[tuple[str, str], ...] = (
    ("Use-case definition and business case", "A R C I I I"),
    ("Data access and data quality", "C A R C I I"),
    ("Solution build and evaluation", "I A R I I I"),
    ("Risk triage and classification", "C R I C A C"),
    ("Responsible AI evaluation", "I A R C C I"),
    ("Go-live approval", "R C C C A C"),
    ("Monitoring and incident handling", "I A R C I I"),
    ("Change management and training", "A R C I I I"),
)
HIGH_RISK_RACI: tuple[tuple[str, str], ...] = (
    ("Human oversight design and reviewer training", "A R C I C C"),
    ("Obligations checklist and compliance documentation", "C R C C I A"),
    ("Fundamental rights impact assessment (where required)", "C C I R I A"),
)
LIMITED_RACI: tuple[tuple[str, str], ...] = (("Transparency notices and content marking", "C A R I I C"),)


def build_raci(tier: str) -> list[tuple[str, dict[str, str]]]:
    """RACI rows for a use case of the given tier. Every activity has exactly one Accountable role."""
    rows = list(BASE_RACI)
    if tier == "high-risk":
        rows += list(HIGH_RISK_RACI) + list(LIMITED_RACI)
    elif tier == "limited":
        rows += list(LIMITED_RACI)
    out = []
    for activity, codes in rows:
        letters = codes.split()
        if letters.count("A") != 1 or len(letters) != len(ROLES):
            raise ValueError(f"invalid RACI row: {activity}")
        out.append((activity, dict(zip(ROLES, letters, strict=True))))
    return out


def raci_markdown(portfolio: Portfolio, tiers: dict[str, TriageResult]) -> str:
    """Render one RACI matrix per distinct tier found in the portfolio."""
    lines = [f"# RACI: {portfolio.name}", "", "R = responsible, A = accountable, C = consulted, I = informed.", ""]
    for tier in ("minimal", "limited", "high-risk"):
        members = [u.id for u in portfolio.use_cases if tiers[u.id].tier == tier]
        if not members:
            continue
        lines += [f"## Tier: {tier}", "", "Applies to: " + ", ".join(members), "",
                  "| Activity | " + " | ".join(ROLES) + " |", "|---|" + "---|" * len(ROLES)]
        lines += [f"| {activity} | " + " | ".join(m[r] for r in ROLES) + " |" for activity, m in build_raci(tier)]
        lines.append("")
    return "\n".join(lines)


def build_reports(portfolio: Portfolio, capacity: int = 2, n_waves: int = 3,
                  fairness: dict[str, FairnessReport] | None = None) -> dict[str, str]:
    """Build all Markdown reports. Returns a mapping of file name to content."""
    scored = score_portfolio(portfolio)
    tiers = triage_portfolio(portfolio)
    planned, excluded = plan_waves(portfolio, scored, tiers, capacity, n_waves)
    return {
        "roadmap.md": roadmap_markdown(portfolio, planned, excluded),
        "risk_register.md": risk_register_markdown(portfolio, build_risk_register(portfolio, tiers, fairness)),
        "raci.md": raci_markdown(portfolio, tiers),
    }
