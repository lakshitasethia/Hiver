"""The escalation decision: an ordered list of transparent rules.

Each rule is ``(name, predicate, decision, reason)``. Rules are evaluated in
order; the **first firing rule wins** and supplies the human-facing reason. The
full trace (every rule's truth value) is returned too, so a reviewer can see not
just why we escalated but also which other rules were close.

Ordering rationale: irreversible / regulated situations first (non-English we
can't read, compliance/legal, money & account-loss intents), then "the model
isn't sure" (low confidence/margin), then softer signals (anger on a medium-risk
issue, weak retrieval, an ungrounded draft). Anything that survives all of them
is safe to auto-send.

Thresholds come from ``config.EscalationConfig`` and were tuned on the dev split
by ``eval/tune_thresholds.py`` — they are not guesses.
"""

from __future__ import annotations

from collections.abc import Callable

from ..config import CONFIG
from ..types import EscalationDecision, GeneratedReply, RuleTrace, Signals

Predicate = Callable[[Signals, "GeneratedReply | None"], bool]

E = CONFIG.escalation


def _low_conf(s: Signals, _r) -> bool:
    return s.clf_confidence < E.low_confidence or s.clf_margin < E.low_margin


RULES: list[tuple[str, Predicate, str, str]] = [
    (
        "non_english",
        lambda s, r: s.is_non_english,
        "escalate",
        "Message does not appear to be in English; a human should handle it.",
    ),
    (
        "compliance_or_legal",
        lambda s, r: bool(s.compliance_hits),
        "escalate",
        "Message raises a legal / regulatory / dispute keyword ({hits}); routing to a human.",
    ),
    (
        "high_risk_intent",
        lambda s, r: s.intent_risk_tier == "high",
        "escalate",
        "Intent '{intent_tier_note}' is high-risk (money, account loss, or security); "
        "auto-replies here are not safe.",
    ),
    (
        "low_model_confidence",
        _low_conf,
        "escalate",
        "Classifier is not confident enough (confidence {conf:.2f}, margin {margin:.2f}); "
        "a human should confirm the intent.",
    ),
    (
        "angry_customer_medium_risk",
        lambda s, r: s.anger_flag and s.intent_risk_tier == "medium",
        "escalate",
        "Customer sentiment is strongly negative on a non-trivial issue; a human touch is warranted.",
    ),
    (
        "weak_retrieval",
        lambda s, r: s.retrieval_max_sim < CONFIG.retrieval.weak_similarity,
        "escalate",
        "No sufficiently similar past conversation to ground a reply "
        "(best match {sim:.2f} < {weak:.2f}).",
    ),
    (
        "ungrounded_draft",
        lambda s, r: r is not None and not r.grounded,
        "escalate",
        "Draft reply introduced specifics not present in the customer's message; needs review.",
    ),
    (
        "pii_present_non_high",
        lambda s, r: bool(s.pii_flags) and s.intent_risk_tier != "low",
        "escalate",
        "Customer shared personal/identifying data ({pii}) on a sensitive issue; handle manually.",
    ),
]


def decide(
    signals: Signals,
    reply: GeneratedReply | None = None,
    *,
    intent_name: str | None = None,
) -> EscalationDecision:
    trace: list[RuleTrace] = []
    winner: tuple[str, str, str] | None = None
    for name, pred, decision, reason in RULES:
        fired = bool(pred(signals, reply))
        trace.append(RuleTrace(name=name, fired=fired))
        if fired and winner is None:
            winner = (name, decision, reason)

    if winner is None:
        return EscalationDecision(
            decision="auto_send",
            reason="All checks passed: confident intent, close historical precedent, "
            "no risk or compliance flags.",
            winning_rule="none",
            trace=trace,
        )

    name, decision, reason_tpl = winner
    reason = reason_tpl.format(
        hits=", ".join(signals.compliance_hits) or "n/a",
        conf=signals.clf_confidence,
        margin=signals.clf_margin,
        sim=signals.retrieval_max_sim,
        weak=CONFIG.retrieval.weak_similarity,
        pii=", ".join(signals.pii_flags) or "n/a",
        intent_tier_note=intent_name or signals.intent_risk_tier,
    )
    return EscalationDecision(decision=decision, reason=reason, winning_rule=name, trace=trace)


def confidence_only_baseline(signals: Signals, threshold: float | None = None) -> EscalationDecision:
    """Baseline #3: escalate iff classifier confidence < threshold. One knob, no rules."""
    threshold = E.low_confidence if threshold is None else threshold
    if signals.clf_confidence < threshold:
        return EscalationDecision(
            decision="escalate",
            reason=f"confidence {signals.clf_confidence:.2f} < {threshold:.2f}",
            winning_rule="confidence_only",
        )
    return EscalationDecision(
        decision="auto_send",
        reason=f"confidence {signals.clf_confidence:.2f} >= {threshold:.2f}",
        winning_rule="confidence_only",
    )
