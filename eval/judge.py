"""LLM-as-judge for generated replies.

The judge scores a candidate reply 1-5 on four axes against a fixed rubric and
returns whether a human should be comfortable sending it. We also run a
**pairwise** comparison against the retrieval baseline (which reply is better,
A/B/tie) with the order randomised to blunt position bias.

Known bias: the judge (Gemini) shares a model family with the generator. We
mitigate — fixed rubric, low temperature, pairwise randomisation, and human
spot-checks recorded in the report — but cannot eliminate it. Stated plainly in
the report's "what the metrics miss" section.
"""

from __future__ import annotations

import json
import random
import re

from support_agent.config import RANDOM_SEED
from support_agent.llm import LLMClient, make_judge_llm

_RUBRIC_SYSTEM = (
    "You are a strict QA reviewer for customer-support replies. Score honestly; "
    "a 5 means you would send it as-is to a paying customer. Respond with one JSON object only."
)

_RUBRIC = """Rate the candidate reply on each axis from 1 (bad) to 5 (excellent):
- helpfulness: does it move the customer's issue forward?
- tone_match: does it fit a professional support voice for this brand?
- factual_caution: does it avoid inventing specifics (order #s, dates, amounts, policy) not given?
Then:
- would_send: true only if you would send it unedited.

Customer message:
\"\"\"{message}\"\"\"

Candidate reply:
\"\"\"{reply}\"\"\"

Return JSON: {{"helpfulness": int, "tone_match": int, "factual_caution": int, "would_send": bool, "rationale": str}}
"""

_PAIRWISE_SYSTEM = (
    "You compare two customer-support replies and pick the better one. "
    "Respond with one JSON object only."
)
_PAIRWISE = """Customer message:
\"\"\"{message}\"\"\"

Reply A:
\"\"\"{a}\"\"\"

Reply B:
\"\"\"{b}\"\"\"

Which reply is better overall for the customer (helpfulness, tone, factual caution)?
Return JSON: {{"winner": "A" | "B" | "tie", "why": str}}
"""


def _parse_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        return json.loads(m.group(0)) if m else {}


class ReplyJudge:
    def __init__(self, llm: LLMClient | None = None, seed: int = RANDOM_SEED):
        self.llm = llm or make_judge_llm()
        self.rng = random.Random(seed)

    def score(self, message: str, reply: str) -> dict:
        if not reply.strip():
            return {"helpfulness": 1, "tone_match": 1, "factual_caution": 3,
                    "would_send": False, "rationale": "empty reply"}
        res = self.llm.complete(
            _RUBRIC.format(message=message, reply=reply),
            system=_RUBRIC_SYSTEM, temperature=0.0, json_mode=True,
        )
        d = _parse_json(res.text)
        for k in ("helpfulness", "tone_match", "factual_caution"):
            d[k] = int(d.get(k, 3))
        d["would_send"] = bool(d.get("would_send", False))
        d["avg"] = round(sum(d[k] for k in ("helpfulness", "tone_match", "factual_caution")) / 3, 3)
        return d

    def pairwise(self, message: str, candidate: str, baseline: str) -> dict:
        swap = self.rng.random() < 0.5
        a, b = (baseline, candidate) if swap else (candidate, baseline)
        res = self.llm.complete(
            _PAIRWISE.format(message=message, a=a, b=b),
            system=_PAIRWISE_SYSTEM, temperature=0.0, json_mode=True,
        )
        d = _parse_json(res.text)
        raw = str(d.get("winner", "tie")).upper()
        if raw == "TIE" or raw not in ("A", "B"):
            outcome = "tie"
        else:
            picked_a = raw == "A"
            candidate_won = (picked_a and not swap) or (not picked_a and swap)
            outcome = "candidate" if candidate_won else "baseline"
        return {"winner": outcome, "why": d.get("why", ""), "order_swapped": swap}
