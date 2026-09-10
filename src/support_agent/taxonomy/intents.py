"""The finalised intent taxonomy (v1).

This is the frozen output of the offline discovery step (``discover.py``): embed a
stratified sample, cluster, read the top terms and nearest messages per cluster,
then a human names and merges them. Ten intents balances granularity against
having enough per-class data to both train and evaluate.

Each intent carries a **risk tier** consumed by the escalation rules:
- ``high``   money movement, account loss, legal/regulatory, security. A wrong
             auto-reply here is expensive or irreversible -> always escalate.
- ``medium`` real problems where a wrong reply is recoverable.
- ``low``    informational or positive; safe to auto-handle when confidence holds.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Intent:
    name: str
    definition: str
    risk: str
    examples: tuple[str, ...]


INTENTS: tuple[Intent, ...] = (
    Intent(
        "order_status",
        "Customer wants the status / ETA / tracking of an existing order or shipment.",
        "low",
        (
            "Where is my order 12345678?",
            "Tracking hasn't updated in 3 days, any news?",
            "When will my package arrive?",
        ),
    ),
    Intent(
        "delivery_issue",
        "Order arrived damaged, incomplete, wrong, or was marked delivered but is missing.",
        "medium",
        (
            "My parcel came smashed and the item is broken.",
            "Says delivered but nothing here.",
            "Got the wrong product in the box.",
        ),
    ),
    Intent(
        "billing_dispute",
        "Customer disputes a charge, wants a refund they believe is owed, or reports incorrect billing.",
        "high",
        (
            "You charged me twice for the same order.",
            "There's a charge on my card I don't recognise.",
            "Promised a refund a week ago and it never came.",
        ),
    ),
    Intent(
        "cancellation",
        "Customer wants to cancel an order, subscription, or account, or asks how to.",
        "high",
        (
            "Cancel my subscription before it renews.",
            "I want to close my account and delete my data.",
            "How do I cancel?",
        ),
    ),
    Intent(
        "account_access",
        "Login, password reset, lockout, 2FA, or suspended-account problems.",
        "high",
        (
            "Can't log in, reset email never arrives.",
            "Locked out after too many attempts.",
            "2FA codes go to my old phone number.",
        ),
    ),
    Intent(
        "technical_bug",
        "Something in the product is broken or misbehaving: crashes, errors, regressions.",
        "medium",
        (
            "App crashes on the payments screen.",
            "Getting error E500 at checkout.",
            "Video stuck at 480p since the update.",
        ),
    ),
    Intent(
        "product_question",
        "Pre-sale or how-to question about features, plans, pricing, or usage. No fault reported.",
        "low",
        (
            "Does the plan include international roaming?",
            "Can I add a second user?",
            "Is there a student discount?",
        ),
    ),
    Intent(
        "complaint_feedback",
        "General dissatisfaction, criticism, or feedback not tied to one resolvable transaction.",
        "medium",
        (
            "Worst support experience I've had.",
            "The new UI is a downgrade.",
            "Been a customer for years and this is how you treat people?",
        ),
    ),
    Intent(
        "praise_thanks",
        "Positive feedback, thanks, or compliments to staff.",
        "low",
        (
            "Your agent was fantastic today.",
            "Love the new dark mode, thank you!",
            "Best service I've had in years.",
        ),
    ),
    Intent(
        "other_unclear",
        "Too short, ambiguous, empty, or off-topic to route confidently.",
        "medium",
        ("hello?", "need help", "DM"),
    ),
)

INTENT_NAMES: tuple[str, ...] = tuple(i.name for i in INTENTS)
RISK_TIER: dict[str, str] = {i.name: i.risk for i in INTENTS}
_BY_NAME: dict[str, Intent] = {i.name: i for i in INTENTS}


def risk_tier(intent: str) -> str:
    return RISK_TIER.get(intent, "medium")


def intent_definitions_block() -> str:
    """Render the taxonomy for an LLM prompt (zero-shot baseline, judge)."""
    lines = []
    for i in INTENTS:
        lines.append(f"- {i.name}: {i.definition}")
    return "\n".join(lines)
