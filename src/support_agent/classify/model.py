"""Logistic-regression intent classifier on sentence embeddings.

Why logistic regression: linear, fast, calibrated-ish probabilities, and its
coefficients are inspectable. On top of frozen sentence embeddings it is a
strong, transparent baseline that trains in seconds and never needs a GPU or an
API key. The model artifact ships with a JSON sidecar recording the training
data hash, label distribution, and cross-validated macro-F1 so a reviewer can
see exactly what produced it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..config import MODELS_DIR, RANDOM_SEED
from ..embeddings import Embedder, make_embedder
from ..types import IntentPrediction

DEFAULT_MODEL_PATH = MODELS_DIR / "clf.joblib"


@dataclass
class TrainReport:
    n_train: int
    n_classes: int
    label_counts: dict[str, int]
    cv_macro_f1: float
    cv_accuracy: float
    embedder: str
    data_sha1: str
    trained_at: str


class IntentClassifier:
    def __init__(self, model, classes: list[str], embedder: Embedder, report: TrainReport | None = None):
        self._model = model
        self.classes = list(classes)
        self._embedder = embedder
        self.report = report

    # -- inference --------------------------------------------------------
    def predict_with_confidence(self, text: str) -> IntentPrediction:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[IntentPrediction]:
        X = self._embedder.encode(texts)
        proba = self._model.predict_proba(X)
        preds = []
        order = np.argsort(-proba, axis=1)
        for i in range(len(texts)):
            dist = {self.classes[j]: float(proba[i, j]) for j in range(len(self.classes))}
            top1, top2 = order[i, 0], order[i, 1] if proba.shape[1] > 1 else order[i, 0]
            preds.append(
                IntentPrediction(
                    intent=self.classes[top1],
                    confidence=float(proba[i, top1]),
                    margin=float(proba[i, top1] - proba[i, top2]),
                    distribution=dist,
                    source="logreg",
                )
            )
        return preds

    # -- persistence ----------------------------------------------------
    def save(self, path: str | Path = DEFAULT_MODEL_PATH) -> None:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self._model, "classes": self.classes}, path)
        if self.report is not None:
            path.with_suffix(".json").write_text(json.dumps(asdict(self.report), indent=2))

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL_PATH, embedder: Embedder | None = None) -> IntentClassifier:
        import joblib

        path = Path(path)
        blob = joblib.load(path)
        report = None
        sidecar = path.with_suffix(".json")
        if sidecar.exists():
            report = TrainReport(**json.loads(sidecar.read_text()))
        emb = embedder or make_embedder()
        if report is not None and report.embedder != emb.name:
            raise RuntimeError(
                f"model {path.name} was trained with embedder '{report.embedder}' but the "
                f"active embedder is '{emb.name}'. Retrain (`make train`) or set "
                f"SUPPORT_AGENT_EMBEDDER to match."
            )
        return cls(blob["model"], blob["classes"], emb, report)


def _sha1_texts(texts: list[str]) -> str:
    h = hashlib.sha1()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def train_classifier(
    texts: list[str],
    labels: list[str],
    *,
    embedder: Embedder | None = None,
    C: float = 1.0,
    seed: int = RANDOM_SEED,
) -> IntentClassifier:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_validate

    if len(texts) != len(labels):
        raise ValueError("texts and labels length mismatch")
    emb = embedder or make_embedder()
    X = emb.encode(texts)
    y = np.array(labels)

    # Newer scikit-learn picks multinomial automatically for multiclass; the old
    # multi_class kwarg was removed.
    model = LogisticRegression(
        C=C, class_weight="balanced", max_iter=2000, random_state=seed
    )

    label_counts = {c: int((y == c).sum()) for c in sorted(set(labels))}
    min_class = min(label_counts.values())
    cv_macro_f1 = cv_acc = float("nan")
    if len(label_counts) > 1 and min_class >= 3:
        folds = min(5, min_class)
        scores = cross_validate(model, X, y, cv=folds, scoring=("f1_macro", "accuracy"))
        cv_macro_f1 = float(np.mean(scores["test_f1_macro"]))
        cv_acc = float(np.mean(scores["test_accuracy"]))

    model.fit(X, y)
    report = TrainReport(
        n_train=len(texts),
        n_classes=len(label_counts),
        label_counts=label_counts,
        cv_macro_f1=round(cv_macro_f1, 4),
        cv_accuracy=round(cv_acc, 4),
        embedder=emb.name,
        data_sha1=_sha1_texts(texts),
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    return IntentClassifier(model, list(model.classes_), emb, report)
