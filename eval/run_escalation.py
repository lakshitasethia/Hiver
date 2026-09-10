"""Escalation eval: the rule engine vs a confidence-threshold-only baseline.

Headline metric is not raw accuracy but **cost-weighted error**: a false
auto-send (we auto-replied when a human was needed) costs 5x a false escalate
(we bothered a human unnecessarily). We also report the plain P/R/F1 on
``should_escalate`` and a decision-curve style sweep of the confidence-only
baseline so the comparison is fair across operating points.
"""

from __future__ import annotations

from support_agent.config import CONFIG
from support_agent.data.normalize import normalize
from support_agent.escalate.rules import confidence_only_baseline, decide
from support_agent.escalate.signals import compute_signals

from .common import LabelRow, binary_report


def _cost(y_true: list[bool], y_pred: list[bool]) -> dict:
    c = CONFIG.escalation
    false_auto = sum(1 for t, p in zip(y_true, y_pred) if t and not p)  # missed escalation
    false_esc = sum(1 for t, p in zip(y_true, y_pred) if not t and p)   # needless escalation
    total = false_auto * c.cost_false_auto_send + false_esc * c.cost_false_escalate
    return {
        "false_auto_send": false_auto,
        "false_escalate": false_esc,
        "weighted_cost": round(total, 2),
        "weighted_cost_per_item": round(total / len(y_true), 3) if y_true else None,
    }


def evaluate(rows: list[LabelRow], classifier=None, retriever=None) -> dict:
    from support_agent.classify.model import IntentClassifier
    from support_agent.data.load import load_sample
    from support_agent.embeddings import make_embedder
    from support_agent.generate.retriever import ReplyRetriever

    emb = make_embedder()
    clf = classifier or IntentClassifier.load(embedder=emb)
    retr = retriever or ReplyRetriever.build(load_sample(), embedder=emb)

    gold = [bool(r.should_escalate) for r in rows]
    rule_pred, conf_pred = [], []
    conf_values = []

    for r in rows:
        clean = normalize(r.text)
        intent = clf.predict_with_confidence(clean.clean)
        exemplars = retr.retrieve(clean.clean, r.brand)
        signals = compute_signals(message=clean, intent=intent, exemplars=exemplars,
                                  history_len=len(r.history or []))
        rule_pred.append(decide(signals, None, intent_name=intent.intent).decision == "escalate")
        conf_pred.append(confidence_only_baseline(signals).decision == "escalate")
        conf_values.append(intent.confidence)

    # sweep the confidence-only baseline threshold for a fair best-case comparison
    sweep = []
    for thr in [x / 100 for x in range(40, 96, 5)]:
        pred = [cv < thr for cv in conf_values]
        rep = binary_report(gold, pred)
        sweep.append({"threshold": thr, "f1": rep["f1"], **_cost(gold, pred)})
    best_conf = min(sweep, key=lambda s: s["weighted_cost"])

    return {
        "n": len(rows),
        "primary_rule_engine": {**binary_report(gold, rule_pred), **_cost(gold, rule_pred)},
        "baseline_confidence_only_default": {
            **binary_report(gold, conf_pred), **_cost(gold, conf_pred),
            "threshold": CONFIG.escalation.low_confidence,
        },
        "baseline_confidence_only_best_sweep": best_conf,
        "sweep": sweep,
    }
