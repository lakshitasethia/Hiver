"""Offline intent discovery — the human-in-the-loop step behind ``intents.py``.

Run ``python -m support_agent.taxonomy.discover`` (or ``make taxonomy``). It:
  1. loads conversations, takes each first customer message,
  2. embeds a stratified sample,
  3. runs K-Means for a range of k and picks k by silhouette score,
  4. prints, per cluster: size, top TF-IDF terms, and the 5 messages nearest the
     centroid.

The output is a *report to read*, not a model artifact. A person uses it to name
and merge clusters into the frozen taxonomy. Re-running it is how we would revise
the taxonomy for a new dataset.
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from ..config import RANDOM_SEED
from ..data.load import iter_first_customer_messages, load_jsonl, load_sample
from ..data.normalize import normalize
from ..embeddings import make_embedder


def _top_terms(texts: list[str], top: int = 8) -> list[str]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    if len(texts) < 2:
        return []
    vec = TfidfVectorizer(stop_words="english", max_features=2000, ngram_range=(1, 2))
    X = vec.fit_transform(texts)
    weights = np.asarray(X.mean(axis=0)).ravel()
    vocab = np.array(vec.get_feature_names_out())
    return vocab[weights.argsort()[::-1][:top]].tolist()


def discover(
    conversations,
    *,
    sample_size: int = 1500,
    k_range: tuple[int, int] = (6, 16),
    seed: int = RANDOM_SEED,
) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    rng = np.random.default_rng(seed)
    rows = [
        (cid, brand, normalize(text).clean)
        for cid, brand, text in iter_first_customer_messages(conversations)
    ]
    rows = [r for r in rows if len(r[2]) > 3]
    if len(rows) > sample_size:
        idx = rng.choice(len(rows), size=sample_size, replace=False)
        rows = [rows[i] for i in idx]
    texts = [r[2] for r in rows]

    emb = make_embedder()
    X = emb.encode(texts)

    best = None
    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)
        sil = silhouette_score(X, km.labels_)
        if best is None or sil > best["silhouette"]:
            best = {"k": k, "silhouette": float(sil), "labels": km.labels_, "centroids": km.cluster_centers_}

    clusters = []
    for c in range(best["k"]):
        members = [i for i, lab in enumerate(best["labels"]) if lab == c]
        if not members:
            continue
        sub = X[members]
        centroid = best["centroids"][c]
        order = np.argsort(-(sub @ centroid))
        nearest = [texts[members[i]] for i in order[:5]]
        brands = Counter(rows[members[i]][1] for i in range(len(members)))
        clusters.append(
            {
                "cluster": c,
                "size": len(members),
                "top_terms": _top_terms([texts[i] for i in members]),
                "nearest": nearest,
                "top_brands": brands.most_common(3),
            }
        )
    clusters.sort(key=lambda d: -d["size"])
    return {"k": best["k"], "silhouette": best["silhouette"], "n": len(texts), "clusters": clusters}


def _format(report: dict) -> str:
    out = [
        f"intent discovery: n={report['n']} messages, chose k={report['k']} "
        f"(silhouette={report['silhouette']:.3f})",
        "",
    ]
    for c in report["clusters"]:
        out.append(f"── cluster {c['cluster']}  (n={c['size']}, brands={c['top_brands']})")
        out.append(f"   terms:  {', '.join(c['top_terms'])}")
        for msg in c["nearest"]:
            out.append(f"   • {msg[:120]}")
        out.append("")
    out.append("Next: name/merge these into src/support_agent/taxonomy/intents.py")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", help="path to a conversations .jsonl (default: sample)")
    ap.add_argument("--sample-size", type=int, default=1500)
    args = ap.parse_args()

    conversations = load_jsonl(args.data) if args.data else load_sample()
    report = discover(conversations, sample_size=args.sample_size)
    print(_format(report))


if __name__ == "__main__":
    main()
