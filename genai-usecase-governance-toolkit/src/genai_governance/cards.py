"""Model / system card generator (Markdown)."""

from __future__ import annotations

from .fairness import FairnessReport, PerformanceMetrics
from .models import UseCase
from .risk_triage import TriageResult


def _fmt(x: float | None, digits: int = 3) -> str:
    return "n/a" if x is None or x != x else f"{x:.{digits}f}"


def system_card(
    use_case: UseCase,
    triage_result: TriageResult,
    performance: PerformanceMetrics | None = None,
    fairness: FairnessReport | None = None,
    model_details: dict[str, str] | None = None,
    evaluation_data_note: str = "",
) -> str:
    """Render a system card for one use case.

    Sections follow the common model card layout: details, intended use, out of
    scope use, risk triage, evaluation, fairness, human oversight, limitations.
    Evaluation sections are only filled from metrics that were actually computed.
    """
    lines = [f"# System card: {use_case.name}", ""]
    lines += ["## System details", ""]
    details = {"Use case id": use_case.id, "Business unit": use_case.business_unit or "n/a",
               "Owner": use_case.owner or "n/a", "Organisation role": use_case.role}
    details.update(model_details or {})
    lines += [f"- {k}: {v}" for k, v in details.items()]
    lines += ["", "## Intended use", "", use_case.description.strip(), ""]
    lines += ["## Out of scope use", "",
              "- Any use outside the business process described above.",
              "- Fully automated decisions with legal or similarly significant effect on individuals "
              "without human review.", ""]
    lines += ["## Risk triage (indicative)", "", f"- Indicative tier: **{triage_result.tier}**"]
    if triage_result.derogation_applied:
        lines.append("- Art. 6(3) derogation assumed; the assessment must be documented (Art. 6(4)).")
    lines += [f"- Reason: {r}" for r in triage_result.reasons]
    lines += ["", "Obligations checklist:", ""]
    lines += [f"- [ ] {o.text} ({o.reference})" for o in triage_result.obligations]
    lines += ["", f"> {triage_result.disclaimer}", ""]

    lines += ["## Evaluation", ""]
    if performance is None:
        lines += ["No evaluation data was supplied for this card.", ""]
    else:
        if evaluation_data_note:
            lines += [f"Evaluation data: {evaluation_data_note}", ""]
        lines += ["| Metric | Value |", "|---|---|",
                  f"| Rows | {performance.n} |",
                  f"| Accuracy | {_fmt(performance.accuracy)} |",
                  f"| Precision | {_fmt(performance.precision)} |",
                  f"| Recall | {_fmt(performance.recall)} |",
                  f"| Specificity | {_fmt(performance.specificity)} |",
                  f"| F1 | {_fmt(performance.f1)} |",
                  f"| Balanced accuracy | {_fmt(performance.balanced_accuracy)} |",
                  f"| ROC AUC | {_fmt(performance.roc_auc)} |",
                  f"| Brier score | {_fmt(performance.brier)} |", ""]

    lines += ["## Fairness", ""]
    if fairness is None:
        lines += ["No fairness evaluation was supplied for this card.", ""]
    else:
        lines += ["| Group | Rows | Selection rate | TPR | FPR | Accuracy |", "|---|---|---|---|---|---|"]
        lines += [f"| {g.group} | {g.n} | {_fmt(g.selection_rate)} | {_fmt(g.tpr)} | {_fmt(g.fpr)} | "
                  f"{_fmt(g.accuracy)} |" for g in fairness.groups]
        lines += ["",
                  f"- Demographic parity difference: {_fmt(fairness.demographic_parity_difference)}",
                  f"- Equalised odds difference: {_fmt(fairness.equalised_odds_difference)}",
                  f"- Disparate impact ratio: {_fmt(fairness.disparate_impact)}"]
        if fairness.dpd_ci95:
            lo, hi = fairness.dpd_ci95
            lines.append(f"- Demographic parity difference, bootstrap 95% interval: [{_fmt(lo)}, {_fmt(hi)}]")
        lines += [""]
        if fairness.flags:
            lines += ["Flags:", ""] + [f"- {f}" for f in fairness.flags] + [""]
        else:
            lines += ["No fairness threshold was exceeded.", ""]

    lines += ["## Human oversight", ""]
    if triage_result.tier == "high-risk":
        lines += ["- A named, trained reviewer confirms every output that affects an individual.",
                  "- Reviewers can override or disregard the output and can stop the system.", ""]
    else:
        lines += ["- Users review generated output before it is used or sent.",
                  "- A feedback channel exists for reporting wrong or harmful output.", ""]
    lines += ["## Limitations", "",
              "- Metrics describe the supplied evaluation file only and do not guarantee behaviour in production.",
              "- Group fairness metrics depend on the chosen sensitive attribute and thresholds.",
              "- The risk tier is an indicative screening result, not a legal classification.", ""]
    return "\n".join(lines)
