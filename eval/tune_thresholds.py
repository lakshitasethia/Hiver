"""Tune escalation thresholds on the DEV split, minimising weighted cost.

Prints recommended ``low_confidence`` / ``low_margin`` / ``weak_similarity`` and
the cost curve. These are the values pasted into ``config.py`` with a ``# tuned``
comment. Kept separate from the headline eval so tuning never touches the test
split.

    python -m eval.tune_thresholds                                   # synthetic dev split
    python -m eval.tune_thresholds --labelset eval/labelset/labels.real.jsonl \
                                   --data data/conversations.jsonl \
                                   --model models/clf.real.joblib     # real dev split
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from support_agent.classify.model import IntentClassifier
from support_agent.config import CONFIG
from support_agent.data.load import load_jsonl, load_sample
from support_agent.data.normalize import normalize
from support_agent.embeddings import make_embedder
from support_agent.escalate.signals import compute_signals
from support_agent.generate.retriever import ReplyRetriever
from support_agent.taxonomy.intents import risk_tier

from .common import LABELSET_PATH, binary_report, load_labelset

E = CONFIG.escalation


def _decide_with(sig, low_conf, low_margin, weak_sim, tier) -> bool:
    """The rule engine, parameterised on the three swept thresholds. The keyword
    rules (severity, compliance, high-risk mass, negative sentiment) are fixed."""
    if sig.is_non_english or sig.compliance_hits or sig.severity_hits or tier == "high":
        return True
    if sig.high_risk_mass >= E.high_risk_mass:
        return True
    if sig.clf_confidence < low_conf or sig.clf_margin < low_margin:
        return True
    if sig.anger_flag and tier == "medium":
        return True
    if sig.sentiment <= E.routine_negative_sentiment and tier in ("low", "medium"):
        return True
    if sig.retrieval_max_sim < weak_sim:
        return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labelset", default=str(LABELSET_PATH))
    ap.add_argument("--data", help="conversations .jsonl for retrieval (default: synthetic sample)")
    ap.add_argument("--model", help="classifier .joblib (default: models/clf.joblib)")
    args = ap.parse_args()

    rows = load_labelset(Path(args.labelset), split="dev")
    if not rows:
        raise SystemExit(f"no dev-split rows in {args.labelset}")

    emb = make_embedder()
    clf = IntentClassifier.load(args.model, embedder=emb) if args.model else IntentClassifier.load(embedder=emb)
    convs = load_jsonl(args.data) if args.data else load_sample()
    retr = ReplyRetriever.build(convs, embedder=emb)

    cache = []
    for r in rows:
        clean = normalize(r.text)
        intent = clf.predict_with_confidence(clean.clean)
        ex = retr.retrieve(clean.clean, r.brand)
        sig = compute_signals(message=clean, intent=intent, exemplars=ex)
        cache.append((bool(r.should_escalate), sig, risk_tier(intent.intent)))

    gold = [g for g, _s, _t in cache]
    best = None
    for lc, lm, ws in itertools.product(
        [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75], [0.05, 0.1, 0.15, 0.2], [0.35, 0.4, 0.45, 0.5]
    ):
        preds = [_decide_with(s, lc, lm, ws, t) for _g, s, t in cache]
        rep = binary_report(gold, preds)
        fa = sum(1 for g, p in zip(gold, preds) if g and not p)
        fe = sum(1 for g, p in zip(gold, preds) if not g and p)
        cost = fa * E.cost_false_auto_send + fe * E.cost_false_escalate
        auto = sum(1 for p in preds if not p)
        if best is None or (cost, -rep["f1"]) < (best["cost"], -best["rep"]["f1"]):
            best = dict(low_confidence=lc, low_margin=lm, weak_similarity=ws, cost=cost,
                        rep=rep, false_auto=fa, false_escalate=fe, auto_rate=auto / len(preds))

    print(f"dev split: {len(rows)} rows from {args.labelset}")
    print("recommended thresholds (minimise weighted cost):")
    for k in ("low_confidence", "low_margin", "weak_similarity"):
        print(f"  {k:16s} = {best[k]}")
    print(f"  -> weighted_cost={best['cost']}  f1={best['rep']['f1']:.3f}  "
          f"auto_send_rate={best['auto_rate']:.2f}  "
          f"(false_auto={best['false_auto']}, false_escalate={best['false_escalate']})")


if __name__ == "__main__":
    main()
