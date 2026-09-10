"""Baseline #1: zero-shot intent classification with an LLM.

No training, no embeddings — just the taxonomy in the prompt. This is the number
the logistic-regression classifier has to beat (on macro-F1, latency, cost, and
determinism) to justify its existence.
"""

from __future__ import annotations

import json
import re

from ..llm import LLMClient, make_llm
from ..taxonomy.intents import INTENT_NAMES, intent_definitions_block
from ..types import IntentPrediction

_SYSTEM = (
    "You are a precise support-ticket triage classifier. "
    "You reply with a single JSON object and nothing else."
)


def _prompt(text: str) -> str:
    return (
        "Classify the customer message into exactly one intent from this list:\n"
        f"{intent_definitions_block()}\n\n"
        f'Customer message:\n"""{text}"""\n\n'
        'Return the single best intent label as JSON: {"intent": "<label>"}. '
        "Return only the label; do not explain."
    )


class ZeroShotLLMClassifier:
    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or make_llm()
        self.source = "llm_zero_shot"

    def predict_with_confidence(self, text: str) -> IntentPrediction:
        res = self.llm.complete(_prompt(text), system=_SYSTEM, temperature=0.0, json_mode=True)
        intent = _parse_label(res.text)
        # Zero-shot gives no probabilities; report a flat, honest placeholder so
        # downstream code that reads confidence still works but cannot be fooled
        # into thinking this is calibrated.
        return IntentPrediction(
            intent=intent,
            confidence=0.5,
            margin=0.0,
            distribution={intent: 1.0},
            source=self.source,
        )

    def predict_batch(self, texts: list[str]) -> list[IntentPrediction]:
        return [self.predict_with_confidence(t) for t in texts]


def _parse_label(raw: str) -> str:
    raw = raw.strip()
    try:
        obj = json.loads(raw)
        cand = str(obj.get("intent", "")).strip()
    except json.JSONDecodeError:
        m = re.search(r'"intent"\s*:\s*"([a-z_]+)"', raw)
        cand = m.group(1) if m else raw.strip().strip('"').lower()
    cand = cand.lower().replace(" ", "_").replace("-", "_")
    return cand if cand in INTENT_NAMES else "other_unclear"
