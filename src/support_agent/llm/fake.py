from __future__ import annotations

import json
import re

from .base import LLMResult

# Keyword -> intent guesses for the fake zero-shot classifier. Intentionally
# crude: the fake exists to make the pipeline runnable and tests deterministic
# offline, not to be a good baseline. The real baseline is GeminiClient.
_INTENT_KEYWORDS = [
    ("where is my order", "order_status"),
    ("tracking", "order_status"),
    ("order status", "order_status"),
    ("hasn't arrived", "delivery_issue"),
    ("damaged", "delivery_issue"),
    ("charged twice", "billing_dispute"),
    ("refund", "billing_dispute"),
    ("cancel", "cancellation"),
    ("can't log in", "account_access"),
    ("password", "account_access"),
    ("crash", "technical_bug"),
    ("error", "technical_bug"),
    ("does it support", "product_question"),
    ("how do i", "product_question"),
    ("worst", "complaint_feedback"),
    ("terrible", "complaint_feedback"),
    ("thank", "praise_thanks"),
    ("love", "praise_thanks"),
]


class FakeLLM:
    """Deterministic stand-in used for tests, CI, and the no-key demo.

    Dispatches on cues in the prompt: judge rubric -> JSON scores,
    classification instruction -> a JSON label, otherwise -> a templated reply
    built from the retrieved exemplar so generation output is still plausible.
    """

    name = "fake"

    def __init__(self, model: str = "fake") -> None:
        self.model = model
        self.name = f"fake:{model}"
        self.calls: list[str] = []

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.3,
        json_mode: bool = False,
    ) -> LLMResult:
        self.calls.append(prompt)
        low = prompt.lower()
        if "helpfulness" in low and "tone_match" in low:
            return self._judge(prompt)
        if "single best intent label" in low or "return only the label" in low:
            return self._classify(prompt)
        return self._generate(prompt)

    # -- task handlers -------------------------------------------------------
    def _classify(self, prompt: str) -> LLMResult:
        low = prompt.lower()
        label = "other_unclear"
        for needle, intent in _INTENT_KEYWORDS:
            if needle in low:
                label = intent
                break
        return LLMResult(text=json.dumps({"intent": label}), model=self.name)

    def _generate(self, prompt: str) -> LLMResult:
        customer = (
            _triple_quoted(prompt)
            or _last_after(prompt, "Customer:")
            or _last_after(prompt, "New message:")
        )
        exemplar = _first_after(prompt, "Agent:")
        brand = _first_after(prompt, "Brand:") or "our team"
        if exemplar:
            reply = (
                f"Thanks for reaching out. {exemplar.strip()} "
                "If anything still looks off, reply here and we'll keep digging."
            )
        else:
            reply = (
                f"Thanks for contacting {brand}. Sorry for the trouble with "
                f'"{(customer or "this").strip()[:80]}". Could you share your order '
                "number or the email on the account so we can look into it right away?"
            )
        return LLMResult(text=reply, model=self.name)

    def _judge(self, prompt: str) -> LLMResult:
        # A slightly generous but stable rubric score so eval plumbing can be
        # exercised without a real judge.
        blocks = _all_triple_quoted(prompt)
        reply = blocks[-1] if blocks else (_last_after(prompt, "Candidate reply:") or "")
        base = 4 if reply and len(reply) > 40 else 3
        grounded = 3 if re.search(r"\b(order #?\d{4,}|tracking \d{6,})\b", reply, re.I) else 4
        payload = {
            "helpfulness": base,
            "tone_match": base,
            "factual_caution": grounded,
            "would_send": base >= 4 and grounded >= 4,
            "rationale": "fake judge: heuristic score based on length and invented-fact check",
        }
        return LLMResult(text=json.dumps(payload), model=self.name)


def _all_triple_quoted(text: str) -> list[str]:
    return [m.strip() for m in re.findall(r'"""(.*?)"""', text, re.S)]


def _triple_quoted(text: str) -> str:
    blocks = _all_triple_quoted(text)
    return blocks[-1] if blocks else ""


def _first_after(text: str, marker: str) -> str:
    idx = text.find(marker)
    if idx == -1:
        return ""
    return text[idx + len(marker):].splitlines()[0].strip()


def _last_after(text: str, marker: str) -> str:
    idx = text.rfind(marker)
    if idx == -1:
        return ""
    return text[idx + len(marker):].splitlines()[0].strip()
