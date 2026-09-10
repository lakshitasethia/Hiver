"""Generate a credential-free sample corpus.

This is **not** meant to fool anyone into thinking it is real Twitter data. It is
a deterministic, templated stand-in so that ``make demo``, ``make train``,
``make eval`` and the whole test suite run for someone who has not downloaded the
Kaggle file. When the real CSV is present, ``make data`` uses that instead and
this module is untouched.

Design goals for the fixture:
- 10 intents, matching ``taxonomy/intents.py``.
- ~6 brands, each with a slightly different sign-off voice.
- A resolved/unresolved mix (~80% resolved) so retrieval has a corpus and the
  ``resolved`` proxy is exercised.
- Seeded rare cases: non-English messages, PII, compliance keywords, sarcasm.
"""

from __future__ import annotations

import random
from dataclasses import asdict

from ..config import DATA_DIR, RANDOM_SEED, SAMPLE_DATA
from ..types import Conversation, Turn

GOLD_INTENTS = DATA_DIR / "sample" / "gold_intents.jsonl"

BRANDS = {
    "NimbusAir": "— The Nimbus Air Team ✈️",
    "BoltMobile": "^JD, BoltMobile Care",
    "FreshCart": "Thanks for shopping with FreshCart!",
    "PixelBank": "PixelBank Support",
    "Streamly": "— Team Streamly",
    "GadgetHub": "//GadgetHub Support",
}

# (intent, customer templates, agent reply templates)
TEMPLATES: dict[str, tuple[list[str], list[str]]] = {
    "order_status": (
        [
            "Where is my order {oid}? It's been {n} days and still nothing.",
            "Any update on order {oid}? Tracking hasn't moved since {day}.",
            "Hi, can you tell me when {oid} will arrive?",
            "still waiting on my package, order {oid}. getting worried",
        ],
        [
            "Thanks for checking in! Order {oid} left our hub {day} and is due in {n} days. Here's the live link: <URL>",
            "Sorry for the wait. {oid} is in transit and should reach you by {day}. Ping us if it slips past that.",
            "I can see {oid} is out for delivery today. You should have it by this evening.",
        ],
    ),
    "delivery_issue": (
        [
            "My parcel arrived damaged. The box was crushed and the screen is cracked.",
            "Package says delivered but it's not here. Order {oid}.",
            "Wrong item in my order {oid} — I got a charger, not the headphones.",
            "Courier left my order in the rain and everything is soaked.",
        ],
        [
            "That's not the unboxing we want for you. I've started a replacement for {oid} at no cost and emailed a prepaid return label.",
            "Sorry about this. I've filed a missing-parcel claim for {oid}; you'll hear back within 2 business days or we reship.",
            "Apologies for the mix-up. Keep the charger — a replacement for the correct item is on its way.",
        ],
    ),
    "billing_dispute": (
        [
            "I was charged twice for the same order {oid}. Want the duplicate refunded now.",
            "There's a {amt} charge on my card I don't recognise.",
            "You promised a refund {n} days ago and I still haven't got it. This is unacceptable.",
            "Why am I still being billed after I downgraded my plan?",
        ],
        [
            "You're right — I see the duplicate on {oid}. I've refunded {amt}; it lands in 3–5 business days.",
            "I've checked the account: that {amt} charge was a pre-auth that should drop off within 48h. If it posts, reply and I'll refund immediately.",
            "Escalating this to our billing team with priority. You'll get an email today with a resolution.",
        ],
    ),
    "cancellation": (
        [
            "I want to cancel my account and delete my data. Done with this service.",
            "Please cancel my subscription before the next renewal on {day}.",
            "How do I cancel? I can't find the option anywhere.",
            "Cancel everything. I've moved to a competitor.",
        ],
        [
            "Sorry to see you go. I've cancelled the renewal effective {day}; you keep access until then. Confirmation sent to your email.",
            "Done — your subscription won't renew. If you'd like your data exported first, say the word.",
            "I can cancel that for you now. Can you confirm the email on the account so I can verify it's you?",
        ],
    ),
    "account_access": (
        [
            "I can't log in. Password reset email never arrives.",
            "Locked out after too many attempts. Need back in today.",
            "Two-factor is sending codes to my old number. Help.",
            "My account says suspended but I don't know why.",
        ],
        [
            "Let's get you back in. I've triggered a fresh reset to the address on file — check spam too. Link is valid for 30 minutes.",
            "I've cleared the lockout. Try again in 5 minutes and let me know if it holds.",
            "I can update the 2FA number once I verify your identity. I'll DM you the secure steps.",
        ],
    ),
    "technical_bug": (
        [
            "The app crashes every time I open the payments tab. Android 14.",
            "Getting error {code} when I try to check out. Cleared cache already.",
            "Video keeps buffering at 480p even on fast wifi since the last update.",
            "Export button does nothing — no file, no error.",
        ],
        [
            "Thanks for the detail. That's a known issue on 14 with the payments tab; a fix ships this week. Workaround: open payments from Settings > Wallet for now.",
            "Error {code} usually means a stale token. Fully sign out and back in — if it repeats, send us the time it happened and we'll pull logs.",
            "Our team is on the buffering regression. Can you share your app version and region so we can match it to the incident?",
        ],
    ),
    "product_question": (
        [
            "Does the {n} GB plan include international roaming?",
            "Can I use this card abroad without fees?",
            "How do I add a second user to my account?",
            "Is there a student discount?",
        ],
        [
            "Good question — the plan includes roaming in 40 countries; a full list is here: <URL>. Outside those it's pay-as-you-go.",
            "No foreign-transaction fees on that card. Just let us know your travel dates so we don't flag the spend.",
            "You can add a user under Settings > Household > Invite. They'll get their own login.",
        ],
    ),
    "complaint_feedback": (
        [
            "This is the worst support experience I've had. Three transfers and no answer.",
            "Your new UI is a downgrade. Everything takes more taps now.",
            "I've been a customer for {n} years and this is how you treat people?",
            "Absolutely terrible. Do better.",
        ],
        [
            "I hear you, and I'm sorry — three transfers is three too many. I'm owning this now; here's my direct line so you don't have to repeat yourself.",
            "That feedback on the UI is fair and I've passed it to the product team with your examples. Thank you for taking the time.",
            "{n} years is a long time and this isn't the standard we want. Let me make the specific issue right — what's outstanding?",
        ],
    ),
    "praise_thanks": (
        [
            "Just want to say your agent Priya was fantastic today. Sorted everything in minutes.",
            "Love the new dark mode. Thank you!",
            "Best customer service I've had in years. Keep it up.",
            "Thanks for the quick refund, really appreciated.",
        ],
        [
            "This made our day — I'll pass it straight to Priya and her manager. Thank you!",
            "So glad you like it! We'll keep the improvements coming.",
            "Thank you for the kind words. We're here whenever you need us.",
        ],
    ),
    "other_unclear": (
        [
            "hello?",
            "is this thing on",
            "need help",
            "DM",
            "🙏🙏🙏",
        ],
        [
            "Hi! We're here. Can you tell us a bit more about what you need help with?",
            "Happy to help — what's going on?",
            "We're listening. What can we do for you today?",
        ],
    ),
}

# Rare cases injected on top of the templated bulk.
NON_ENGLISH = [
    "Bonjour, ma commande {oid} n'est jamais arrivée. Que se passe-t-il ?",
    "Hola, me cobraron dos veces por el pedido {oid}. Quiero un reembolso.",
    "Mein Paket wurde beschädigt geliefert. Ich möchte einen Ersatz.",
]
COMPLIANCE = [
    "If this isn't fixed today I'm filing a chargeback and contacting my lawyer.",
    "Under GDPR I am requesting all data you hold on me and its deletion.",
    "This is my final notice before I report you to the regulator.",
]
SARCASM = [
    "Oh great, another 'we're looking into it'. Fantastic. Love that for me.",
    "Wow, only 45 minutes on hold. New record. Truly world class.",
]


def _fill(t: str, rng: random.Random) -> str:
    return t.format(
        oid=f"{rng.randint(10000000, 99999999)}",
        n=rng.randint(2, 9),
        day=rng.choice(["Monday", "Tuesday", "Friday", "yesterday", "last week"]),
        amt=f"${rng.randint(5, 240)}.{rng.randint(0, 99):02d}",
        code=f"E{rng.randint(100, 999)}",
    )


def build_corpus(n: int = 620, seed: int = RANDOM_SEED) -> list[tuple[Conversation, str]]:
    """Return ``(conversation, true_intent)`` pairs.

    The true intent is written to ``gold_intents.jsonl`` and used only to *seed*
    the hand-labelled eval set and to sanity-check weak labelling in tests. The
    training pipeline never reads it — it relies on weak supervision, exactly as
    it must for the real, unlabelled Kaggle data.
    """
    rng = random.Random(seed)
    intents = list(TEMPLATES)
    convs: list[tuple[Conversation, str]] = []
    for i in range(n):
        intent = intents[i % len(intents)] if i < len(intents) * 3 else rng.choice(intents)
        brand = rng.choice(list(BRANDS))
        cust_t, agent_t = TEMPLATES[intent]
        cust = _fill(rng.choice(cust_t), rng)

        # inject rare variants
        if intent in ("order_status", "delivery_issue", "billing_dispute") and rng.random() < 0.05:
            cust = _fill(rng.choice(NON_ENGLISH), rng)
        elif intent in ("billing_dispute", "cancellation", "complaint_feedback") and rng.random() < 0.10:
            cust = cust + " " + rng.choice(COMPLIANCE)
        elif intent == "complaint_feedback" and rng.random() < 0.20:
            cust = rng.choice(SARCASM)

        turns = [Turn("customer", cust)]
        # ~15% get a clarifying back-and-forth
        if rng.random() < 0.15:
            turns.append(Turn("agent", "Thanks — can you share your order number or account email?"))
            turns.append(Turn("customer", _fill("It's {oid}, email is jo@example.com", rng)))

        resolved = rng.random() < 0.82
        if resolved:
            reply = _fill(rng.choice(agent_t), rng)
            turns.append(Turn("agent", f"{reply} {BRANDS[brand]}"))
        else:
            turns.append(Turn("agent", "Looking into this now, back to you shortly."))
            turns.append(Turn("customer", "Any update?? still waiting"))

        convs.append(
            (
                Conversation(conv_id=f"syn-{i:04d}", brand=brand, turns=turns, resolved=resolved),
                intent,
            )
        )
    rng.shuffle(convs)
    return convs


def write_sample(path=SAMPLE_DATA, n: int = 620) -> int:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    pairs = build_corpus(n=n)
    with open(path, "w") as fh:
        for conv, _intent in pairs:
            fh.write(json.dumps(asdict(conv), ensure_ascii=False) + "\n")
    with open(GOLD_INTENTS, "w") as fh:
        for conv, intent in pairs:
            fh.write(json.dumps({"conv_id": conv.conv_id, "intent": intent}) + "\n")
    return len(pairs)


if __name__ == "__main__":
    count = write_sample()
    print(f"wrote {count} synthetic conversations to {SAMPLE_DATA}")
    print(f"wrote {count} gold intents to {GOLD_INTENTS}")
