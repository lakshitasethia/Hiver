"""Classification eval: logistic-regression classifier vs LLM zero-shot baseline.

Metrics: macro-F1, accuracy, per-class P/R/F1, confusion matrix, and accuracy
within confidence buckets (a calibration check — does higher confidence actually
mean higher accuracy?).
"""

from __future__ import annotations

from support_agent.classify.baseline_llm import ZeroShotLLMClassifier
from support_agent.classify.model import IntentClassifier
from support_agent.data.normalize import normalize
from support_agent.taxonomy.intents import INTENT_NAMES

from .common import LabelRow, multiclass_report


def _confidence_buckets(rows, preds) -> list[dict]:
    edges = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)]
    out = []
    for lo, hi in edges:
        idx = [i for i, p in enumerate(preds) if lo <= p.confidence < hi]
        if not idx:
            out.append({"range": f"{lo:.2f}-{hi:.2f}", "n": 0, "accuracy": None})
            continue
        acc = sum(1 for i in idx if preds[i].intent == rows[i].intent) / len(idx)
        out.append({"range": f"{lo:.2f}-{hi:.2f}", "n": len(idx), "accuracy": round(acc, 4)})
    return out


def evaluate(rows: list[LabelRow], *, run_llm_baseline: bool = True, classifier=None) -> dict:
    texts = [normalize(r.text).clean for r in rows]
    gold = [r.intent for r in rows]

    clf = classifier or IntentClassifier.load()
    clf_preds = clf.predict_batch(texts)
    clf_report = multiclass_report(gold, [p.intent for p in clf_preds], list(INTENT_NAMES))
    clf_report["confidence_buckets"] = _confidence_buckets(rows, clf_preds)
    clf_report["model_sidecar"] = clf.report.__dict__ if clf.report else None

    result = {"n": len(rows), "primary": clf_report}

    # trivial baseline: always predict the most common intent in the eval set
    from collections import Counter

    majority = Counter(gold).most_common(1)[0][0]
    result["baseline_trivial_majority"] = {
        **multiclass_report(gold, [majority] * len(gold), list(INTENT_NAMES)),
        "predicts": majority,
    }

    if run_llm_baseline:
        zs = ZeroShotLLMClassifier()
        zs_pred, zs_gold, n_failed = [], [], 0
        for text, g in zip(texts, gold):
            try:
                zs_pred.append(zs.predict_with_confidence(text).intent)
                zs_gold.append(g)
            except Exception as exc:  # noqa: BLE001 - skip a flaky call, keep the run alive
                n_failed += 1
                print(f"  [zero-shot] {type(exc).__name__}: {exc}")
        if zs_pred:
            result["baseline_llm_zero_shot"] = multiclass_report(
                zs_gold, zs_pred, list(INTENT_NAMES)
            )
            result["baseline_llm_zero_shot"]["n_failed"] = n_failed
            result["delta_macro_f1"] = round(
                clf_report["macro_f1"] - result["baseline_llm_zero_shot"]["macro_f1"], 4
            )
    return result
