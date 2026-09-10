"""Run every eval suite and render the results into ``docs/report.md``.

The report prose is written by hand; only the numbers between the
``<!-- BEGIN:results -->`` / ``<!-- END:results -->`` markers are generated, so
the headline figures can never drift from the code that produced them.
Run: ``python -m eval.report`` (or ``make report``).
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import platform

from support_agent.config import CONFIG

from . import run_classification, run_escalation, run_generation
from .common import REPO_ROOT, load_labelset, save_json

REPORT_MD = REPO_ROOT / "docs" / "report.md"
BEGIN = "<!-- BEGIN:results -->"
END = "<!-- END:results -->"


def _md_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _render(cls: dict, gen: dict, esc: dict, meta: dict) -> str:
    p = cls["primary"]
    judge_note = (
        f" · judge=`{meta['judge_llm']}`" if meta.get("judge_llm") and meta["judge_llm"] != meta["llm"] else ""
    )
    parts = [f"_Generated {meta['generated_at']} · {meta['n_labelset']} labelled examples "
             f"· embedder=`{meta['embedder']}` · llm=`{meta['llm']}`{judge_note}_", ""]

    # Classification
    parts.append("### Classification\n")
    base = cls.get("baseline_llm_zero_shot")
    parts.append(_md_table(
        ["metric", "logreg (ours)", "LLM zero-shot (baseline)", "delta"],
        [
            ["macro-F1", p["macro_f1"], base["macro_f1"] if base else "n/a",
             cls.get("delta_macro_f1", "n/a")],
            ["accuracy", p["accuracy"], base["accuracy"] if base else "n/a", ""],
        ],
    ))
    parts.append("\n**Per-class F1 (ours):**\n")
    parts.append(_md_table(
        ["intent", "precision", "recall", "f1", "support"],
        [[k, v["precision"], v["recall"], v["f1"], v["support"]]
         for k, v in p["per_class"].items()],
    ))
    parts.append("\n**Accuracy by confidence bucket (calibration check):**\n")
    parts.append(_md_table(
        ["confidence range", "n", "accuracy"],
        [[b["range"], b["n"], b["accuracy"]] for b in p["confidence_buckets"]],
    ))

    # Generation
    parts.append("\n### Generation\n")
    if gen.get("n_gen_failed") or gen.get("n_judge_failed"):
        parts.append(
            f"_{gen.get('n_generated', gen['n'])}/{gen['n']} replies generated; "
            f"{gen.get('n_gen_failed', 0)} generation + {gen.get('n_judge_failed', 0)} judge "
            f"calls dropped (API errors)._\n"
        )
    a = gen["automated"]
    parts.append(_md_table(
        ["metric", "RAG (ours)", "retrieval-only (baseline)"],
        [
            ["semantic sim to historical reply",
             a["rag_semantic_sim_to_history"], a["baseline_semantic_sim_to_history"]],
            ["groundedness rate (no invented specifics)", a["rag_groundedness_rate"], "1.0 (verbatim)"],
            ["length ratio vs historical reply", a["rag_len_ratio_vs_history"], "1.0"],
        ],
    ))
    if "llm_judge" in gen:
        j = gen["llm_judge"]
        parts.append("\n**LLM-judge (1–5; would_send is a rate):**\n")
        parts.append(_md_table(
            ["helpfulness", "tone_match", "factual_caution", "would_send_rate"],
            [[j["avg_helpfulness"], j["avg_tone_match"], j["avg_factual_caution"],
              j["would_send_rate"]]],
        ))
        pw = j["pairwise_vs_retrieval_baseline"]
        parts.append(f"\nPairwise vs retrieval baseline — RAG better: **{pw['candidate']}**, "
                     f"baseline better: {pw['baseline']}, tie: {pw['tie']}.")

    # Escalation
    parts.append("\n### Escalation\n")
    r = esc["primary_rule_engine"]
    b0 = esc["baseline_confidence_only_default"]
    b1 = esc["baseline_confidence_only_best_sweep"]
    parts.append(_md_table(
        ["metric", "rule engine (ours)", "confidence-only @default", "confidence-only @best sweep"],
        [
            ["precision", r["precision"], b0["precision"], "—"],
            ["recall", r["recall"], b0["recall"], "—"],
            ["f1", r["f1"], b0["f1"], b1["f1"]],
            ["false auto-sends", r["false_auto_send"], b0["false_auto_send"], b1["false_auto_send"]],
            ["false escalations", r["false_escalate"], b0["false_escalate"], b1["false_escalate"]],
            ["weighted cost (5x/1x)", r["weighted_cost"], b0["weighted_cost"], b1["weighted_cost"]],
            ["auto-send rate", r["auto_send_rate"], b0["auto_send_rate"], b1.get("auto_send_rate", "—")],
        ],
    ))
    parts.append(
        f"\nCost model: a false auto-send costs {CONFIG.escalation.cost_false_auto_send}x a false "
        "escalation. Auto-send rate is shown because a policy that escalates everything scores a "
        "great cost while automating nothing."
    )

    return "\n".join(parts)


def run_and_render(
    *,
    limit: int | None = None,
    use_judge: bool | None = None,
    write_md: bool = True,
    labelset_path=None,
    data_path: str | None = None,
    model_path: str | None = None,
) -> dict:
    from pathlib import Path

    rows = load_labelset(Path(labelset_path)) if labelset_path else load_labelset()
    if limit:
        rows = rows[:limit]
    use_judge = (os.getenv("SUPPORT_AGENT_LLM", "auto") != "fake") if use_judge is None else use_judge

    from support_agent.classify.model import IntentClassifier
    from support_agent.data.load import load_jsonl, load_sample
    from support_agent.embeddings import make_embedder
    from support_agent.generate.retriever import ReplyRetriever
    from support_agent.llm import make_judge_llm, make_llm

    emb = make_embedder()
    # Shared objects, built once. Defaults = the committed synthetic path.
    conversations = load_jsonl(data_path) if data_path else load_sample()
    classifier = (
        IntentClassifier.load(model_path, embedder=emb)
        if model_path
        else IntentClassifier.load(embedder=emb)
    )
    retriever = ReplyRetriever.build(conversations, embedder=emb)

    cls = run_classification.evaluate(rows, run_llm_baseline=True, classifier=classifier)
    gen = run_generation.evaluate(rows, use_judge=True, conversations=conversations)
    esc = run_escalation.evaluate(rows, classifier=classifier, retriever=retriever)

    meta = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "n_labelset": len(rows),
        "labelset": str(labelset_path) if labelset_path else "synthetic",
        "data": data_path or "synthetic sample",
        "embedder": make_embedder().name,
        "llm": make_llm().name,
        "judge_llm": make_judge_llm().name,
        "python": platform.python_version(),
    }
    payload = {"meta": meta, "classification": cls, "generation": gen, "escalation": esc}
    summary_path = save_json("eval_summary.json", payload)

    if not write_md:
        print(f"wrote {summary_path} (skipped {REPORT_MD.name})")
        return payload

    block = f"{BEGIN}\n{_render(cls, gen, esc, meta)}\n{END}"
    if REPORT_MD.exists():
        text = REPORT_MD.read_text()
        if BEGIN in text and END in text:
            pre = text.split(BEGIN)[0]
            post = text.split(END)[1]
            REPORT_MD.write_text(pre + block + post)
        else:
            REPORT_MD.write_text(text.rstrip() + "\n\n## Results\n\n" + block + "\n")
    else:
        REPORT_MD.write_text("# Evaluation Results\n\n" + block + "\n")
    print(f"wrote {REPORT_MD} and {summary_path}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-report", action="store_true",
                    help="write eval/results/*.json only; leave docs/report.md untouched")
    ap.add_argument("--labelset", help="path to a labels .jsonl (default: the synthetic set)")
    ap.add_argument("--data", help="conversations .jsonl for the retrieval corpus (default: synthetic sample)")
    ap.add_argument("--model", help="classifier .joblib (default: models/clf.joblib)")
    args = ap.parse_args()
    run_and_render(
        limit=args.limit,
        write_md=not args.no_report,
        labelset_path=args.labelset,
        data_path=args.data,
        model_path=args.model,
    )


if __name__ == "__main__":
    main()
