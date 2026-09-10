"""Tune escalation thresholds on the DEV split, minimising weighted cost.

Prints a recommended ``low_confidence`` / ``low_margin`` / ``weak_similarity``
and the cost curve. These are the values pasted into ``config.py`` with a
``# tuned`` comment. Kept separate from the headline eval so tuning never touches
the test split.
"""

from __future__ import annotations

import itertools

from support_agent.classify.model import IntentClassifier
from support_agent.config import CONFIG
from support_agent.data.load import load_sample
from support_agent.data.normalize import normalize
from support_agent.embeddings import make_embedder
from support_agent.escalate.signals import compute_signals
from support_agent.generate.retriever import ReplyRetriever
from support_agent.taxonomy.intents import risk_tier

from .common import binary_report, load_labelset


def _decide_with(signals, low_conf, low_margin, weak_sim, intent_tier) -> bool:
    if signals.is_non_english or signals.compliance_hits or intent_tier == "high":
        return True
    if signals.clf_confidence < low_conf or signals.clf_margin < low_margin:
        return True
    if signals.anger_flag and intent_tier == "medium":
        return True
    if signals.retrieval_max_sim < weak_sim:
        return True
    return False


def main() -> None:
    rows = load_labelset(split="dev")
    if not rows:
        raise SystemExit("no dev-split rows in the label set; run `make labelset` first")

    emb = make_embedder()
    clf = IntentClassifier.load(embedder=emb)
    retr = ReplyRetriever.build(load_sample(), embedder=emb)

    cache = []
    for r in rows:
        clean = normalize(r.text)
        intent = clf.predict_with_confidence(clean.clean)
        ex = retr.retrieve(clean.clean, r.brand)
        sig = compute_signals(message=clean, intent=intent, exemplars=ex)
        cache.append((bool(r.should_escalate), sig, risk_tier(intent.intent)))

    c = CONFIG.escalation
    best = None
    for lc, lm, ws in itertools.product(
        [0.45, 0.5, 0.55, 0.6, 0.65], [0.05, 0.1, 0.15, 0.2], [0.35, 0.4, 0.45, 0.5]
    ):
        preds = [_decide_with(sig, lc, lm, ws, tier) for _g, sig, tier in cache]
        gold = [g for g, _s, _t in cache]
        rep = binary_report(gold, preds)
        fa = sum(1 for g, p in zip(gold, preds) if g and not p)
        fe = sum(1 for g, p in zip(gold, preds) if not g and p)
        cost = fa * c.cost_false_auto_send + fe * c.cost_false_escalate
        if best is None or (cost, -rep["f1"]) < (best["cost"], -best["rep"]["f1"]):
            best = {"low_confidence": lc, "low_margin": lm, "weak_similarity": ws,
                    "cost": cost, "rep": rep, "false_auto": fa, "false_escalate": fe}

    print("recommended thresholds (dev split, minimise weighted cost):")
    for k in ("low_confidence", "low_margin", "weak_similarity"):
        print(f"  {k:16s} = {best[k]}")
    print(f"  -> weighted_cost={best['cost']}  f1={best['rep']['f1']}  "
          f"(false_auto={best['false_auto']}, false_escalate={best['false_escalate']})")


if __name__ == "__main__":
    main()
