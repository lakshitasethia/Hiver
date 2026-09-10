"""RAG reply generation: few-shot the retrieved exemplars into the LLM.

The prompt is deliberately strict about *groundedness*: match the brand's tone
from the examples, but never invent an order number, date, amount, or policy the
customer did not provide. After generation we run cheap heuristics to check the
draft did not hallucinate concrete facts; a failure sets ``grounded=False`` which
the escalation layer treats as a reason to send the message to a human.
"""

from __future__ import annotations

import re

from ..config import CONFIG
from ..llm import LLMClient, make_llm
from ..types import Exemplar, GeneratedReply

_SYSTEM = (
    "You are a customer-support agent drafting a reply in the brand's established "
    "voice. Rules: (1) match the tone and sign-off style of the example replies; "
    "(2) never invent specifics — order numbers, tracking numbers, dates, refund "
    "amounts, or policy details not present in the customer's message or the "
    "examples; if a specific is needed, ask for it; (3) one concise reply, no "
    "preamble, no markdown."
)

_INVENTED_FACT = re.compile(
    r"\border\s*#?\s*\d{4,}\b|\btracking\s*#?\s*\w{6,}\b|\$\d+(?:\.\d{2})?\b|"
    r"\b\d{1,2}[:/]\d{2}\b|\b(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{1,2}\b",
    re.I,
)


def _fewshot_block(exemplars: list[Exemplar]) -> str:
    parts = []
    for i, ex in enumerate(exemplars, 1):
        tag = ex.brand + ("" if ex.same_brand else " [other brand]")
        parts.append(
            f"Example {i} (brand: {tag}, similarity {ex.similarity:.2f})\n"
            f"Customer: {ex.customer_text}\n"
            f"Agent: {ex.agent_text}"
        )
    return "\n\n".join(parts)


def _check_grounded(reply: str, customer_text: str) -> tuple[bool, list[str]]:
    notes: list[str] = []
    for m in _INVENTED_FACT.finditer(reply):
        span = m.group(0)
        if span.lower() not in customer_text.lower():
            notes.append(f"reply contains specific not in customer message: {span!r}")
    return (not notes), notes


class ReplyGenerator:
    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or make_llm()

    def generate(
        self, message: str, brand: str, exemplars: list[Exemplar], *, intent: str | None = None
    ) -> GeneratedReply:
        if not exemplars:
            # Nothing to ground on: return a safe clarifying holding reply and
            # let escalation decide. Do not call the LLM with no context.
            return GeneratedReply(
                text=(
                    "Thanks for reaching out — I want to get this right. Could you "
                    "share your order number or the email on your account so I can "
                    "look into it?"
                ),
                exemplars=[],
                used_llm=False,
                model="none",
                grounded=True,
                groundedness_notes=["no exemplars; returned clarifying reply"],
            )

        prompt = (
            f"Brand: {brand}\n"
            + (f"Detected intent: {intent}\n" if intent else "")
            + "Draft a reply to the NEW message, using the examples only for tone "
            "and structure.\n\n"
            f"{_fewshot_block(exemplars)}\n\n"
            f"NEW message:\nCustomer: {message}\n\nYour reply:"
        )
        res = self.llm.complete(prompt, system=_SYSTEM, temperature=CONFIG.llm.temperature)
        text = res.text.strip().strip('"')
        grounded, notes = _check_grounded(text, message)
        return GeneratedReply(
            text=text,
            exemplars=exemplars,
            used_llm=True,
            model=res.model,
            grounded=grounded,
            groundedness_notes=notes,
        )
