"""Train the intent classifier from weakly-labelled data.

Pipeline: load conversations -> take first customer message -> weak-label with
keyword rules (``classify.weak_label``) -> drop abstains -> train logistic
regression on embeddings -> save model + JSON sidecar.

``python -m support_agent.train`` uses the committed sample; pass ``--data`` for
a real ``conversations.jsonl`` produced by ``scripts/download_data.sh``.
"""

from __future__ import annotations

import argparse
from collections import Counter

from .classify.model import DEFAULT_MODEL_PATH, train_classifier
from .classify.weak_label import weak_label
from .config import ensure_dirs
from .data.load import iter_first_customer_messages, load_jsonl, load_sample
from .data.normalize import normalize
from .embeddings import make_embedder


def build_training_set(conversations, *, min_confidence: float = 0.6):
    texts: list[str] = []
    labels: list[str] = []
    abstain = 0
    for _cid, _brand, raw in iter_first_customer_messages(conversations):
        clean = normalize(raw)
        if not clean.is_english:
            continue
        intent, conf, _fired = weak_label(clean.clean)
        if intent is None or conf < min_confidence:
            abstain += 1
            continue
        texts.append(clean.clean)
        labels.append(intent)
    return texts, labels, abstain


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", help="conversations .jsonl (default: committed sample)")
    ap.add_argument("--out", default=str(DEFAULT_MODEL_PATH))
    ap.add_argument("--min-confidence", type=float, default=0.6)
    ap.add_argument("--C", type=float, default=1.0)
    args = ap.parse_args()

    ensure_dirs()
    conversations = load_jsonl(args.data) if args.data else load_sample()
    texts, labels, abstain = build_training_set(
        conversations, min_confidence=args.min_confidence
    )
    if len(set(labels)) < 2:
        raise SystemExit(
            f"weak labelling produced <2 classes ({Counter(labels)}); "
            "check the data or lower --min-confidence"
        )

    print(
        f"loaded {len(conversations)} conversations -> "
        f"{len(texts)} weakly-labelled training rows ({abstain} abstained)"
    )
    print("label distribution:", dict(Counter(labels).most_common()))

    clf = train_classifier(texts, labels, embedder=make_embedder(), C=args.C)
    clf.save(args.out)
    r = clf.report
    print(
        f"\nsaved -> {args.out}\n"
        f"  embedder      : {r.embedder}\n"
        f"  classes       : {r.n_classes}\n"
        f"  CV macro-F1   : {r.cv_macro_f1}\n"
        f"  CV accuracy   : {r.cv_accuracy}\n"
        f"  data sha1     : {r.data_sha1[:12]}"
    )


if __name__ == "__main__":
    main()
