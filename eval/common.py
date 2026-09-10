"""Shared helpers for the eval suites: label-set IO and metric primitives."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LABELSET_PATH = REPO_ROOT / "eval" / "labelset" / "labels.jsonl"
RESULTS_DIR = REPO_ROOT / "eval" / "results"


@dataclass
class LabelRow:
    id: str
    brand: str
    text: str
    intent: str                      # gold
    should_escalate: bool            # gold
    escalation_reason: str = ""      # gold, free text
    history: list[str] | None = None
    ideal_reply_notes: str = ""
    notes: str = ""
    split: str = "test"              # "dev" for threshold tuning, "test" for headline


def load_labelset(path: Path = LABELSET_PATH, *, split: str | None = None) -> list[LabelRow]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `make labelset` (synthetic) or follow "
            "docs/labeling-guide.md to build it from the real data."
        )
    rows: list[LabelRow] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        rows.append(LabelRow(**d))
    if split:
        rows = [r for r in rows if r.split == split]
    return rows


def save_json(name: str, payload: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = RESULTS_DIR / name
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return p


# -- metric primitives (kept dependency-light and explicit) -----------------
def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def binary_report(y_true: list[bool], y_pred: list[bool]) -> dict:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    p, r, f1 = prf(tp, fp, fn)
    n = len(y_true)
    return {
        "n": n,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "accuracy": round((tp + tn) / n, 4) if n else 0.0,
    }


def multiclass_report(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    per_class = {}
    macro_f1 = 0.0
    for c in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        p, r, f1 = prf(tp, fp, fn)
        support = sum(1 for t in y_true if t == c)
        per_class[c] = {"precision": round(p, 4), "recall": round(r, 4),
                        "f1": round(f1, 4), "support": support}
        macro_f1 += f1
    macro_f1 /= len(labels) if labels else 1
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) if y_true else 0.0
    # confusion matrix as nested dict
    cm = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        if t in cm and p in cm[t]:
            cm[t][p] += 1
    return {
        "n": len(y_true),
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(acc, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
    }
