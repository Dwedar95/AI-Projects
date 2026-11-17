"""Rule-based, indicative risk triage oriented on the EU AI Act (Regulation (EU) 2024/1689).

This module is a simplified screening aid for discovery workshops. It is not
legal advice and it does not replace an assessment by qualified counsel. It
encodes a subset of the regulation as yes/no questions, does not model every
exception, and does not encode application dates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

DISCLAIMER = (
    "Indicative triage only. This is a simplified aid based on a subset of Regulation (EU) 2024/1689 "
    "(EU AI Act). It is not legal advice, does not cover every exception or application date, and "
    "must be confirmed by qualified legal counsel before any decision is taken."
)

Tier = Literal["prohibited", "high-risk", "limited", "minimal"]
TIER_ORDER: dict[str, int] = {"minimal": 0, "limited": 1, "high-risk": 2, "prohibited": 3}


@dataclass(frozen=True)
class Question:
    """One yes/no screening question with the reference it relies on."""

    id: str
    text: str
    tier: Tier
    reference: str
    keywords: tuple[str, ...] = ()


QUESTIONS: tuple[Question, ...] = (
    # Prohibited practices, Article 5(1).
    Question("subliminal_manipulation", "Does the system use subliminal, manipulative or deceptive techniques "
             "that materially distort behaviour and are likely to cause significant harm?", "prohibited",
             "Art. 5(1)(a)", ("subliminal", "manipulat")),
    Question("exploits_vulnerabilities", "Does it exploit vulnerabilities due to age, disability or social or "
             "economic situation in a way likely to cause significant harm?", "prohibited", "Art. 5(1)(b)"),
    Question("social_scoring", "Does it evaluate or classify people based on social behaviour or personal "
             "traits, leading to detrimental treatment in unrelated contexts or disproportionate treatment?",
             "prohibited", "Art. 5(1)(c)", ("social scoring", "social score", "citizen score")),
    Question("individual_crime_prediction", "Does it predict the risk of a person committing a crime based "
             "solely on profiling or personality traits?", "prohibited", "Art. 5(1)(d)",
             ("predictive policing", "crime prediction")),
    Question("untargeted_face_scraping", "Does it build facial recognition databases through untargeted "
             "scraping of facial images from the internet or CCTV?", "prohibited", "Art. 5(1)(e)",
             ("face scraping", "scrape facial", "scraping facial")),
    Question("emotion_recognition_work_education", "Does it infer emotions of people in the workplace or in "
             "education (other than for medical or safety reasons)?", "prohibited", "Art. 5(1)(f)",
             ("employee emotion", "emotions of employees", "student emotion", "emotions of students")),
    Question("biometric_categorisation_sensitive", "Does it categorise individuals from biometric data to infer "
             "sensitive attributes such as race, political opinions, religion or sexual orientation?",
             "prohibited", "Art. 5(1)(g)"),
    Question("realtime_remote_biometric_id_law_enforcement", "Is it real-time remote biometric identification "
             "in publicly accessible spaces for law enforcement?", "prohibited", "Art. 5(1)(h)"),
    # High-risk, Article 6.
    Question("safety_component_regulated_product", "Is it a safety component of, or itself, a product covered "
             "by EU harmonisation legislation that requires third-party conformity assessment (for example "
             "machinery, medical devices)?", "high-risk", "Art. 6(1), Annex I",
             ("medical device", "safety component")),
    Question("biometrics", "Is it used for remote biometric identification, biometric categorisation or "
             "emotion recognition?", "high-risk", "Art. 6(2), Annex III point 1",
             ("biometric", "facial recognition", "face recognition", "emotion recognition")),
    Question("critical_infrastructure", "Is it a safety component in the management or operation of critical "
             "infrastructure (digital infrastructure, road traffic, water, gas, heating, electricity)?",
             "high-risk", "Art. 6(2), Annex III point 2", ("critical infrastructure", "power grid", "water supply")),
    Question("education", "Does it determine access or admission to education, evaluate learning outcomes, or "
             "monitor students during tests?", "high-risk", "Art. 6(2), Annex III point 3",
             ("admission", "exam", "grade students", "grades students", "proctoring")),
    Question("employment", "Is it used for recruitment or selection, or for decisions on promotion, "
             "termination, task allocation or monitoring and evaluation of workers?", "high-risk",
             "Art. 6(2), Annex III point 4",
             ("recruit", "cv screening", "screen cv", "candidate ranking", "rank candidates", "hiring",
              "applicant", "promotion decision", "employee performance", "workforce monitoring")),
    Question("essential_services", "Does it evaluate eligibility for public benefits, creditworthiness or "
             "credit scores, risk and pricing in life or health insurance, or triage emergency calls?",
             "high-risk", "Art. 6(2), Annex III point 5",
             ("credit scor", "creditworthiness", "loan approval", "loan decision", "insurance pricing",
              "insurance premium", "benefit eligibility", "emergency call")),
    Question("law_enforcement", "Is it used by or on behalf of law enforcement, for example to assess "
             "evidence reliability or the risk of offending?", "high-risk", "Art. 6(2), Annex III point 6",
             ("law enforcement", "police")),
    Question("migration_border", "Is it used in migration, asylum or border control management?", "high-risk",
             "Art. 6(2), Annex III point 7", ("asylum", "border control", "visa application")),
    Question("justice_democracy", "Does it assist judicial authorities in researching and interpreting facts "
             "and law, or is it intended to influence elections or voting behaviour?", "high-risk",
             "Art. 6(2), Annex III point 8", ("court ruling", "judicial", "election", "voter")),
    # Transparency obligations, Article 50.
    Question("interacts_with_people", "Does the system interact directly with natural persons (for example a "
             "chatbot or voice assistant)?", "limited", "Art. 50(1)",
             ("chatbot", "virtual assistant", "voice assistant", "copilot", "conversational", "assistant")),
    Question("generates_synthetic_content", "Does it generate synthetic text, audio, image or video content?",
             "limited", "Art. 50(2)",
             ("generat", "drafting", "drafts", "summar", "content creation")),
    Question("deep_fake", "Does it generate or manipulate image, audio or video content that resembles real "
             "people, places or events (deep fake)?", "limited", "Art. 50(4)", ("deep fake", "deepfake", "avatar")),
    Question("public_interest_text", "Does it generate text published to inform the public on matters of "
             "public interest without human review and editorial responsibility?", "limited", "Art. 50(4)"),
)

# Conditions of the Article 6(3) derogation and the profiling exclusion.
DEROGATION_QUESTIONS: tuple[Question, ...] = (
    Question("narrow_procedural_task", "Does the system only perform a narrow procedural task?", "minimal",
             "Art. 6(3)(a)"),
    Question("improves_prior_human_activity", "Does it only improve the result of a previously completed "
             "human activity?", "minimal", "Art. 6(3)(b)"),
    Question("detects_patterns_no_replacement", "Does it only detect decision-making patterns or deviations "
             "without replacing or influencing the human assessment without proper human review?", "minimal",
             "Art. 6(3)(c)"),
    Question("preparatory_task", "Does it only perform a preparatory task to an assessment?", "minimal",
             "Art. 6(3)(d)"),
    Question("profiling_of_persons", "Does the system perform profiling of natural persons?", "high-risk",
             "Art. 6(3), last subparagraph", ("profiling", "profile individuals")),
)

CONTEXT_QUESTIONS: tuple[Question, ...] = (
    Question("public_body_or_public_service", "Is the deployer a body governed by public law or a private "
             "entity providing public services?", "high-risk", "Art. 27(1)"),
    Question("decisions_about_persons", "Does the system make or assist decisions about natural persons?",
             "high-risk", "Art. 26(11)"),
    Question("used_in_workplace", "Will the system be put into service at the workplace, affecting workers?",
             "high-risk", "Art. 26(7)"),
)

ALL_QUESTION_IDS = {q.id for q in QUESTIONS + DEROGATION_QUESTIONS + CONTEXT_QUESTIONS}

PROVIDER_HIGH_RISK = (
    ("Establish and maintain a risk management system across the lifecycle", "Art. 9"),
    ("Apply data governance to training, validation and test data", "Art. 10"),
    ("Draw up and keep technical documentation", "Art. 11"),
    ("Enable automatic record-keeping (logging) of events", "Art. 12"),
    ("Provide transparency and instructions for use to deployers", "Art. 13"),
    ("Design for effective human oversight", "Art. 14"),
    ("Achieve appropriate accuracy, robustness and cybersecurity", "Art. 15"),
    ("Operate a quality management system", "Art. 17"),
    ("Complete the applicable conformity assessment before placing on the market", "Art. 43"),
    ("Draw up the EU declaration of conformity and affix the CE marking", "Art. 47, Art. 48"),
    ("Register the system in the EU database", "Art. 49"),
    ("Run post-market monitoring and report serious incidents", "Art. 72, Art. 73"),
)

DEPLOYER_HIGH_RISK = (
    ("Use the system in accordance with the instructions for use", "Art. 26(1)"),
    ("Assign human oversight to competent, trained persons with authority", "Art. 26(2)"),
    ("Ensure input data under your control is relevant and sufficiently representative", "Art. 26(4)"),
    ("Monitor operation and inform the provider about risks and serious incidents", "Art. 26(5)"),
    ("Keep automatically generated logs for at least six months", "Art. 26(6)"),
)


@dataclass
class Obligation:
    """One checklist item with its reference."""

    text: str
    reference: str


@dataclass
class TriageResult:
    """Indicative tier with the reasons and obligations checklist."""

    tier: Tier
    reasons: list[str]
    obligations: list[Obligation]
    inferred_from_description: list[str] = field(default_factory=list)
    derogation_applied: bool = False
    disclaimer: str = DISCLAIMER


def infer_answers(description: str) -> dict[str, bool]:
    """Suggest positive answers from keywords in a free-text description.

    Only non-prohibited and unambiguous prohibited keywords are matched. The
    result is a pre-fill for a workshop, explicit answers always override it.
    """
    text = description.lower()
    hits: dict[str, bool] = {}
    for q in QUESTIONS + DEROGATION_QUESTIONS:
        for kw in q.keywords:
            if re.search(r"\b" + re.escape(kw), text):
                hits[q.id] = True
                break
    return hits


def triage(description: str = "", answers: dict[str, bool] | None = None, role: str = "deployer") -> TriageResult:
    """Map a use case to an indicative tier.

    ``answers`` are explicit yes/no answers keyed by question id. Keyword
    matches in ``description`` fill unanswered questions only. Precedence:
    prohibited, then high-risk (unless the Article 6(3) derogation applies),
    then limited (transparency), otherwise minimal.
    """
    explicit = dict(answers or {})
    unknown = set(explicit) - ALL_QUESTION_IDS
    if unknown:
        raise ValueError(f"unknown triage question ids: {sorted(unknown)}")
    if role not in ("provider", "deployer"):
        raise ValueError("role must be 'provider' or 'deployer'")
    inferred = {k: v for k, v in infer_answers(description).items() if k not in explicit}
    merged = {**inferred, **explicit}

    def source(qid: str) -> str:
        return "keyword match in description" if qid in inferred else "answered yes"

    by_tier: dict[str, list[Question]] = {"prohibited": [], "high-risk": [], "limited": []}
    for q in QUESTIONS:
        if merged.get(q.id):
            by_tier[q.tier].append(q)

    reasons: list[str] = []
    obligations: list[Obligation] = []
    derogation = False
    literacy = Obligation("Ensure a sufficient level of AI literacy of staff operating the system", "Art. 4")

    if by_tier["prohibited"]:
        tier: Tier = "prohibited"
        reasons = [f"{q.id} ({q.reference}, {source(q.id)}): {q.text}" for q in by_tier["prohibited"]]
        obligations = [Obligation("Do not place on the market, put into service or use this practice", "Art. 5")]
    elif by_tier["high-risk"]:
        annex_iii_only = all(q.id != "safety_component_regulated_product" for q in by_tier["high-risk"])
        derogation_hits = [q for q in DEROGATION_QUESTIONS[:4] if merged.get(q.id)]
        profiling = bool(merged.get("profiling_of_persons"))
        if annex_iii_only and derogation_hits and not profiling:
            derogation = True
            tier = "limited" if by_tier["limited"] else "minimal"
            reasons = [f"{q.id} ({q.reference}, {source(q.id)}): Annex III area touched" for q in by_tier["high-risk"]]
            reasons += [f"derogation condition met: {q.id} ({q.reference})" for q in derogation_hits]
            reasons.append("no profiling of natural persons, so the Art. 6(3) derogation may apply")
            obligations.append(Obligation(
                "Document the assessment that the system is not high-risk before placing it on the market "
                "or putting it into service", "Art. 6(4)"))
            obligations.append(Obligation("Register the system assessed as not high-risk in the EU database", "Art. 49(2)"))
        else:
            tier = "high-risk"
            reasons = [f"{q.id} ({q.reference}, {source(q.id)}): {q.text}" for q in by_tier["high-risk"]]
            if derogation_hits and profiling:
                reasons.append("Art. 6(3) derogation not available: the system performs profiling of natural persons")
            pairs = PROVIDER_HIGH_RISK if role == "provider" else DEPLOYER_HIGH_RISK
            obligations = [Obligation(t, r) for t, r in pairs]
            if role == "deployer":
                if merged.get("used_in_workplace"):
                    obligations.append(Obligation(
                        "Inform workers' representatives and affected workers before putting the system "
                        "into service at the workplace", "Art. 26(7)"))
                if merged.get("decisions_about_persons") or merged.get("employment") or merged.get("essential_services"):
                    obligations.append(Obligation(
                        "Inform natural persons that they are subject to the use of a high-risk AI system",
                        "Art. 26(11)"))
                if merged.get("public_body_or_public_service") or merged.get("essential_services"):
                    obligations.append(Obligation(
                        "Check whether a fundamental rights impact assessment is required and perform it "
                        "before first use", "Art. 27"))
    elif by_tier["limited"]:
        tier = "limited"
    else:
        tier = "minimal"
        reasons = ["no prohibited practice, high-risk area or transparency trigger was identified"]

    if tier != "prohibited":
        for q in by_tier["limited"]:
            if tier == "limited":
                reasons.append(f"{q.id} ({q.reference}, {source(q.id)}): {q.text}")
            obligations.append(Obligation(_TRANSPARENCY_TEXT[q.id], q.reference))
        obligations.append(literacy)

    return TriageResult(
        tier=tier,
        reasons=reasons,
        obligations=obligations,
        inferred_from_description=sorted(inferred),
        derogation_applied=derogation,
    )


_TRANSPARENCY_TEXT = {
    "interacts_with_people": "Inform people that they are interacting with an AI system unless this is obvious",
    "generates_synthetic_content": "Mark generated content as artificially generated in a machine-readable "
                                   "format (obligation on the provider of the generating system)",
    "deep_fake": "Disclose that the content has been artificially generated or manipulated",
    "public_interest_text": "Disclose that published text was artificially generated, unless it had human "
                            "review and editorial responsibility",
}
