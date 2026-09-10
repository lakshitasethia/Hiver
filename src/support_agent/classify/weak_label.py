"""Weak supervision for classifier training labels.

The Kaggle data has no intent labels and hand-labelling 3M tweets is off the
table. So we build a **noisy** training set with keyword/regex labelling
functions (Snorkel-style, but hand-rolled to keep dependencies light), train the
classifier on it, and reserve human labelling for the *evaluation* set only.

Each labelling function votes for at most one intent. Votes are combined by
priority (money/access/cancellation cues beat generic ones) then by count.
``weak_label`` returns ``(intent | None, confidence, fired_rules)``; ``None`` is
an abstain and those rows are dropped from training.
"""

from __future__ import annotations

import re

from ..taxonomy.intents import INTENT_NAMES

# (intent, priority, compiled pattern). Higher priority wins ties.
_RULES: list[tuple[str, int, re.Pattern]] = [
    ("billing_dispute", 5, re.compile(r"charged twice|double charge|duplicate charge|don'?t recognise|dont recognize|unrecognis|chargeback|refund (?:never|still|not).{0,20}(?:came|arrived|received)|wrong(?:ly)? billed|still being billed", re.I)),
    ("billing_dispute", 3, re.compile(r"\brefund\b|\binvoice\b|\bovercharged\b|billing", re.I)),
    ("cancellation", 5, re.compile(r"cancel (?:my|the) (?:account|subscription|order|plan|membership)|close my account|delete my (?:account|data)|how do i cancel|want to cancel", re.I)),
    ("account_access", 5, re.compile(r"can'?t log ?in|cannot log ?in|locked out|password reset|reset (?:email|link)|2fa|two[- ]factor|account (?:is )?suspended|verification code", re.I)),
    ("delivery_issue", 5, re.compile(r"arrived damaged|came (?:broken|smashed|crushed)|box was (?:crushed|damaged)|marked delivered|says delivered|wrong item|missing item|left (?:it )?in the rain|never (?:arrived|delivered)", re.I)),
    ("order_status", 4, re.compile(r"where('?s| is) my (?:order|package|parcel|delivery|stuff)|track(?:ing)?(?: number| update| info|'?s stuck)?|when (?:will|does|is) .{0,30}(?:arrive|deliver|ship|come)|status (?:of|on) (?:my )?order|hasn'?t (?:moved|updated|arrived)|still (?:waiting|wadng) (?:on|for) my (?:order|package|delivery|parcel)|any update on (?:my )?order|eta on", re.I)),
    ("technical_bug", 4, re.compile(r"\bcrash(?:es|ing|ed)?\b|\bbug\b|error (?:code )?[a-z]?\d{2,}|\bglitch\b|not working since (?:the )?update|won'?t load|keeps buffering|freezes", re.I)),
    ("product_question", 3, re.compile(r"\bdoes (?:it|the plan|this) (?:include|support|have|work)\b|how do i (?:add|use|set up|enable)|is there a (?:discount|free trial|student)|can i use .* abroad|what'?s the difference between", re.I)),
    ("praise_thanks", 4, re.compile(r"\b(thank you|thanks|much appreciated|appreciate it)\b|\b(love|loving) (?:the|your|this)\b|best (?:customer )?service|was (?:fantastic|amazing|brilliant|great|wonderful)|made my day", re.I)),
    ("complaint_feedback", 3, re.compile(r"\bworst\b|\bterrible\b|\bawful\b|\bunacceptable\b|do better|disappointed|downgrade|this is how you treat|been a customer for \d+ years", re.I)),
]

_SHORT_UNCLEAR = re.compile(r"^\s*(hi|hey|hello|help|\?+|dm|pm|\W+)\s*$", re.I)


def weak_label(text: str) -> tuple[str | None, float, list[str]]:
    if not text or _SHORT_UNCLEAR.match(text) or len(text.split()) <= 2:
        # A greeting or a two-word fragment is a confident "not routable yet".
        return "other_unclear", 0.75, ["short_or_greeting"]

    votes: dict[str, int] = {}
    prio: dict[str, int] = {}
    fired: list[str] = []
    for intent, priority, pat in _RULES:
        if pat.search(text):
            votes[intent] = votes.get(intent, 0) + 1
            prio[intent] = max(prio.get(intent, 0), priority)
            fired.append(f"{intent}:{priority}")

    if not votes:
        return None, 0.0, []

    # winner: highest priority, then most votes
    winner = max(votes, key=lambda i: (prio[i], votes[i]))
    total = sum(votes.values())
    agreement = votes[winner] / total
    confidence = min(0.95, 0.55 + 0.1 * prio[winner] / 5 + 0.3 * agreement)
    return winner, round(confidence, 3), fired


def weak_label_batch(
    texts: list[str], *, min_confidence: float = 0.0
) -> list[tuple[str | None, float, list[str]]]:
    out = []
    for t in texts:
        intent, conf, fired = weak_label(t)
        if intent is not None and conf < min_confidence:
            intent = None
        out.append((intent, conf, fired))
    return out


def coverage(texts: list[str]) -> dict:
    labelled = 0
    per_intent: dict[str, int] = {k: 0 for k in INTENT_NAMES}
    for t in texts:
        intent, _c, _f = weak_label(t)
        if intent:
            labelled += 1
            per_intent[intent] += 1
    return {
        "n": len(texts),
        "labelled": labelled,
        "coverage": round(labelled / max(1, len(texts)), 3),
        "per_intent": per_intent,
    }
