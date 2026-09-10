"""Does the LLM-as-judge agree with a human?

`eval/judge_agreement_pairs.jsonl` holds 25 (message, reply) pairs — real RAG
drafts for AmazonHelp messages — each carrying a **human** score
(`human_helpfulness`, `human_tone_match`, `human_factual_caution` on 1–5, and
`human_would_send` bool) that I assigned by reading the pair against the same
rubric the LLM judge uses (`eval/judge.py`).

This script re-scores the same pairs with the LLM judge and reports agreement:

  - would_send: exact-match rate + Cohen's kappa (chance-corrected)
  - rubric axes: mean absolute error + Pearson r on the per-pair average

Run: `python -m eval.judge_agreement`
"""

from __future__ import annotations

import json
from pathlib import Path

from .common import RESULTS_DIR, save_json
from .judge import ReplyJudge

PAIRS = Path(__file__).resolve().parent / "judge_agreement_pairs.jsonl"
AXES = ("helpfulness", "tone_match", "factual_caution")


def _kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    if not n:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return (po - pe) / (1 - pe) if pe != 1 else 1.0


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def main() -> None:
    if not PAIRS.exists():
        raise SystemExit(f"{PAIRS} not found — generate it first (see module docstring).")
    pairs = [json.loads(line) for line in PAIRS.read_text().splitlines() if line.strip()]
    scored = [p for p in pairs if "human_would_send" in p]
    if len(scored) < len(pairs):
        raise SystemExit(
            f"{len(pairs) - len(scored)} pairs are missing human_* scores; "
            "fill them in before running."
        )

    judge = ReplyJudge()
    print(f"judge model: {judge.llm.name}  ·  {len(scored)} human-scored pairs\n")

    h_ws, m_ws = [], []
    h_avg, m_avg = [], []
    axis_ae: dict[str, list[float]] = {ax: [] for ax in AXES}
    n_fail = 0
    for p in scored:
        try:
            m = judge.score(p["message"], p["reply"])
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            print(f"  [judge] {p['id']}: {type(exc).__name__}: {exc}")
            continue
        h_ws.append(bool(p["human_would_send"]))
        m_ws.append(bool(m["would_send"]))
        for ax in AXES:
            axis_ae[ax].append(abs(p[f"human_{ax}"] - m[ax]))
        h_avg.append(sum(p[f"human_{ax}"] for ax in AXES) / 3)
        m_avg.append(sum(m[ax] for ax in AXES) / 3)

    n = len(h_ws)
    result = {
        "judge_model": judge.llm.name,
        "n": n,
        "n_judge_failed": n_fail,
        "would_send_exact_match": round(sum(1 for a, b in zip(h_ws, m_ws) if a == b) / n, 3),
        "would_send_cohen_kappa": round(_kappa(h_ws, m_ws), 3),
        "human_would_send_rate": round(sum(h_ws) / n, 3),
        "judge_would_send_rate": round(sum(m_ws) / n, 3),
        "rubric_avg_pearson_r": round(_pearson(h_avg, m_avg), 3),
        "rubric_axis_mean_abs_error": {ax: round(sum(v) / len(v), 3) for ax, v in axis_ae.items()},
    }
    print(json.dumps(result, indent=2))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_json("judge_agreement.json", result)
    print(f"\nwrote {RESULTS_DIR / 'judge_agreement.json'}")


if __name__ == "__main__":
    main()
