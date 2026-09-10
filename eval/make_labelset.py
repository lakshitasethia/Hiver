"""Build the evaluation label set.

**Synthetic path (default):** the sample generator records the true intent of
every conversation, so we can emit a fully-labelled set directly — stratified by
intent and brand, held out from the classifier's weak-label training pool.
``should_escalate`` gold is derived from a small, explicit policy (high-risk
intent, compliance keyword, or non-English -> escalate) and is meant to be
reviewed by hand.

**Real path:** run with ``--data data/conversations.jsonl --unlabelled``. It
samples the same way but leaves ``intent``/``should_escalate`` blank for a human
to fill following ``docs/labeling-guide.md``.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from support_agent.config import RANDOM_SEED
from support_agent.data.load import load_jsonl, load_sample
from support_agent.data.normalize import normalize
from support_agent.data.synthetic import GOLD_INTENTS
from support_agent.escalate.signals import _compliance_hits  # explicit policy inputs
from support_agent.taxonomy.intents import risk_tier

LABELSET_PATH = Path(__file__).resolve().parent / "labelset" / "labels.jsonl"


def _gold_escalation(intent: str, text: str, is_english: bool) -> tuple[bool, str]:
    if not is_english:
        return True, "non-English message"
    hits = _compliance_hits(text)
    if hits:
        return True, f"compliance/legal keyword: {', '.join(hits)}"
    if risk_tier(intent) == "high":
        return True, f"high-risk intent ({intent})"
    return False, "routine, low/medium risk"


def build_synthetic(target: int, seed: int) -> list[dict]:
    convs = {c.conv_id: c for c in load_sample()}
    gold = {
        json.loads(line)["conv_id"]: json.loads(line)["intent"]
        for line in GOLD_INTENTS.read_text().splitlines() if line.strip()
    }
    by_intent: dict[str, list[str]] = defaultdict(list)
    for cid, intent in gold.items():
        by_intent[intent].append(cid)

    rng = random.Random(seed)
    per_intent = max(3, target // len(by_intent))
    rows: list[dict] = []
    for intent, cids in sorted(by_intent.items()):
        rng.shuffle(cids)
        for cid in cids[:per_intent]:
            c = convs[cid]
            text = c.first_customer_text
            clean = normalize(text)
            esc, reason = _gold_escalation(intent, text, clean.is_english)
            rows.append({
                "id": cid,
                "brand": c.brand,
                "text": text,
                "intent": intent,
                "should_escalate": esc,
                "escalation_reason": reason,
                "history": [t.text for t in c.turns[1:-1]] or None,
                "ideal_reply_notes": "",
                "notes": "auto-seeded from synthetic generator truth; review before trusting",
                "split": "dev" if rng.random() < 0.3 else "test",
            })
    rng.shuffle(rows)
    return rows[:target] if target < len(rows) else rows


def build_unlabelled(data_path: str, target: int, seed: int) -> list[dict]:
    convs = load_jsonl(data_path)
    rng = random.Random(seed)
    rng.shuffle(convs)
    rows = []
    for c in convs[: target * 2]:
        text = c.first_customer_text
        if not text or len(text.split()) < 3:
            continue
        rows.append({
            "id": c.conv_id, "brand": c.brand, "text": text,
            "intent": "", "should_escalate": None, "escalation_reason": "",
            "history": [t.text for t in c.turns[1:-1]] or None,
            "ideal_reply_notes": "", "notes": "TO LABEL", "split": "test",
        })
        if len(rows) >= target:
            break
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", help="real conversations .jsonl")
    ap.add_argument("--unlabelled", action="store_true", help="emit blank labels for hand-labelling")
    ap.add_argument("--target", type=int, default=200)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--out", default=str(LABELSET_PATH))
    args = ap.parse_args()

    if args.unlabelled:
        if not args.data:
            raise SystemExit("--unlabelled needs --data")
        rows = build_unlabelled(args.data, args.target, args.seed)
    else:
        rows = build_synthetic(args.target, args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_dev = sum(1 for r in rows if r["split"] == "dev")
    print(f"wrote {len(rows)} rows to {out}  (dev={n_dev}, test={len(rows) - n_dev})")


if __name__ == "__main__":
    main()
