"""Pure functions that turn a message + model outputs into escalation signals.

Every signal is cheap, deterministic, and independently testable. The rule engine
(``rules.py``) consumes the resulting :class:`Signals` object; keeping extraction
separate from policy means we can re-tune thresholds without touching detection
and vice versa.
"""

from __future__ import annotations

import re

from ..config import CONFIG
from ..taxonomy.intents import risk_tier
from ..types import CleanMessage, Exemplar, IntentPrediction, Signals

# Tiny polarity lexicon. Not trying to compete with a trained sentiment model —
# it is a transparent, explainable stand-in and its blind spot (sarcasm) is one
# of the documented failure modes.
_NEG = {
    "angry", "furious", "unacceptable", "terrible", "awful", "worst", "hate",
    "disgusted", "ridiculous", "appalling", "useless", "scam", "fraud", "never",
    "again", "still", "waiting", "ignored", "rude", "disappointed", "frustrated",
    "annoyed", "outrageous", "horrible", "pathetic", "joke", "nonsense",
}
_POS = {
    "thanks", "thank", "great", "awesome", "love", "excellent", "perfect",
    "amazing", "brilliant", "wonderful", "fantastic", "appreciate", "helpful",
    "best", "happy", "pleased",
}
_INTENSIFIERS = {"very", "so", "absolutely", "completely", "utterly", "extremely"}

_PII = {
    "email": re.compile(r"<EMAIL>|[\w.+-]+@[\w-]+\.[\w.-]+"),
    "phone": re.compile(r"<PHONE>|(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)"),
    "card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "order_id": re.compile(r"<ORDERID>|\border\s*#?\s*[A-Z0-9]{5,}\b", re.I),
}

_COMPLIANCE = [
    "chargeback", "charge back", "gdpr", "ccpa", "data protection", "right to be forgotten",
    "lawyer", "legal action", "sue", "lawsuit", "solicitor", "attorney",
    "regulator", "ombudsman", "trading standards", "ftc", "consumer protection",
    "delete my data", "delete my account", "close my account", "cancel my account",
    "police", "fraud department", "dispute the charge",
]

_ANGER_PUNCT = re.compile(r"[!?]{2,}|[A-Z]{4,}")

# Severity cues that make a message escalation-worthy regardless of the predicted
# intent or the (weak) sentiment score. Added after the real-data eval showed the
# rule engine auto-sending "you all stole my package", "out of date chicken", and
# a store-safety complaint because the lexicon read them as neutral.
_SEVERITY = {
    "safety": re.compile(
        r"\b(unsafe|not safe|feel safe|dangerous|hazard|injur|hurt|sick|ill|poison|"
        r"out of date|expired|rotten|mould|mold|contaminat|allergic|choking)\b", re.I
    ),
    "theft_loss": re.compile(
        r"\b(stole|stolen|theft|robbed|never (arrived|received|got|came|delivered)|"
        r"missing|disappeared|lost my|taken without)\b", re.I
    ),
    "repeated_failure": re.compile(
        r"\b(third time|3rd time|fourth time|again and again|every (day|time|morning)|"
        r"still (waiting|not|no|haven'?t|hasn'?t)|for (\d+|several|many) (days|weeks|months)|"
        r"\d+ (days|weeks) (and|now|later)|told by \d+|multiple (times|reps|agents)|"
        r"no one (has )?(responded|replied|helped|called)|keep (getting|being))\b", re.I
    ),
    "demand": re.compile(
        r"\b(contact me immediately|call me (now|immediately|back)|speak to (a )?(manager|"
        r"supervisor|human)|need someone (to|from)|this is (unacceptable|ridiculous)|"
        r"final notice|last (chance|warning)|escalate this)\b", re.I
    ),
}


def severity_hits(text: str) -> list[str]:
    return [name for name, pat in _SEVERITY.items() if pat.search(text)]


def high_risk_mass(distribution: dict[str, float]) -> float:
    """Total probability the classifier put on *any* high-risk intent."""
    return sum(p for name, p in distribution.items() if risk_tier(name) == "high")


def sentiment_score(text: str) -> float:
    tokens = re.findall(r"[a-z']+", text.lower())
    if not tokens:
        return 0.0
    score = 0.0
    for i, tok in enumerate(tokens):
        mult = 1.5 if i and tokens[i - 1] in _INTENSIFIERS else 1.0
        if tok in _NEG:
            score -= mult
        elif tok in _POS:
            score += mult
    if _ANGER_PUNCT.search(text):
        score -= 0.5
    return max(-1.0, min(1.0, score / max(3, len(tokens) ** 0.5)))


def _pii_flags(text: str) -> list[str]:
    return [name for name, pat in _PII.items() if pat.search(text)]


def _compliance_hits(text: str) -> list[str]:
    low = text.lower()
    return [kw for kw in _COMPLIANCE if kw in low]


def compute_signals(
    *,
    message: CleanMessage,
    intent: IntentPrediction,
    exemplars: list[Exemplar],
    history_len: int = 0,
) -> Signals:
    raw = message.raw
    sent = sentiment_score(raw)
    max_sim = max((e.similarity for e in exemplars), default=0.0)
    return Signals(
        clf_confidence=intent.confidence,
        clf_margin=intent.margin,
        intent_risk_tier=risk_tier(intent.intent),
        sentiment=sent,
        anger_flag=sent <= CONFIG.escalation.anger_sentiment,
        retrieval_max_sim=max_sim,
        pii_flags=_pii_flags(raw),
        compliance_hits=_compliance_hits(raw),
        severity_hits=severity_hits(raw),
        high_risk_mass=round(high_risk_mass(intent.distribution), 4),
        is_non_english=not message.is_english,
        history_len=history_len,
    )
