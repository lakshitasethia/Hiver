"""Generation eval: RAG reply vs retrieval-only baseline.

Automated: semantic similarity to the brand's historical reply for that
conversation, length ratio, and a groundedness rate (fraction of drafts that did
not introduce unseen specifics). LLM-judge: rubric scores + pairwise win-rate vs
the baseline.
"""

from __future__ import annotations

import numpy as np

from support_agent.data.load import load_sample
from support_agent.embeddings import make_embedder
from support_agent.generate.baseline_retrieval import nearest_reply_baseline
from support_agent.generate.generator import ReplyGenerator
from support_agent.generate.retriever import ReplyRetriever
from support_agent.llm import make_llm

from .common import LabelRow
from .judge import ReplyJudge


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) or 1.0))


def evaluate(
    rows: list[LabelRow],
    *,
    use_judge: bool = True,
    reference_by_id: dict[str, str] | None = None,
    conversations=None,
) -> dict:
    emb = make_embedder()
    convs = conversations if conversations is not None else load_sample()
    retriever = ReplyRetriever.build(convs, embedder=emb)
    gen = ReplyGenerator(make_llm())
    judge = ReplyJudge() if use_judge else None  # ReplyJudge picks its own (possibly different) model
    ref_map = reference_by_id or {c.conv_id: c.last_agent_text for c in convs}

    rag_sims, base_sims = [], []
    rag_grounded = 0
    n_generated = 0
    len_ratios = []
    judge_rows = []
    pairwise = {"candidate": 0, "baseline": 0, "tie": 0}
    n_gen_failed = n_judge_failed = 0

    for r in rows:
        exemplars = retriever.retrieve(r.text, r.brand)
        base = nearest_reply_baseline(exemplars)
        try:
            rag = gen.generate(r.text, r.brand, exemplars, intent=r.intent)
        except Exception as exc:  # noqa: BLE001 - one flaky API call shouldn't kill the run
            n_gen_failed += 1
            print(f"  [gen] {r.id}: {type(exc).__name__}: {exc}")
            continue
        n_generated += 1
        ref = ref_map.get(r.id, "")

        if ref:
            v = emb.encode([ref, rag.text or " ", base.text or " "])
            rag_sims.append(_cos(v[0], v[1]))
            base_sims.append(_cos(v[0], v[2]))
            len_ratios.append(len(rag.text) / max(1, len(ref)))
        rag_grounded += int(rag.grounded)

        if judge is not None:
            try:
                s = judge.score(r.text, rag.text)
                judge_rows.append(s)
                if base.text.strip():
                    pw = judge.pairwise(r.text, rag.text, base.text)
                    pairwise[pw["winner"]] += 1
            except Exception as exc:  # noqa: BLE001
                n_judge_failed += 1
                print(f"  [judge] {r.id}: {type(exc).__name__}: {exc}")

    def _mean(xs):
        return round(float(np.mean(xs)), 4) if xs else None

    out = {
        "n": len(rows),
        "n_generated": n_generated,
        "n_gen_failed": n_gen_failed,
        "n_judge_failed": n_judge_failed,
        "automated": {
            "rag_semantic_sim_to_history": _mean(rag_sims),
            "baseline_semantic_sim_to_history": _mean(base_sims),
            "rag_groundedness_rate": round(rag_grounded / n_generated, 4) if n_generated else None,
            "rag_len_ratio_vs_history": _mean(len_ratios),
        },
    }
    if judge is not None:
        out["llm_judge"] = {
            "avg_helpfulness": _mean([s["helpfulness"] for s in judge_rows]),
            "avg_tone_match": _mean([s["tone_match"] for s in judge_rows]),
            "avg_factual_caution": _mean([s["factual_caution"] for s in judge_rows]),
            "would_send_rate": round(
                sum(s["would_send"] for s in judge_rows) / len(judge_rows), 4
            ) if judge_rows else None,
            "pairwise_vs_retrieval_baseline": pairwise,
        }
    return out
