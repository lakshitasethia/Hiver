"""Interactive labeller for the evaluation set.

Reads a seed file of rows (from ``make_labelset --unlabelled``), shows one
message at a time, and records ``intent`` + ``should_escalate`` from single
keystrokes. Resumable: re-run and it skips rows already labelled in the output.

    python -m eval.label_cli --in  eval/labelset/labels.real.unlabelled.jsonl \
                             --out eval/labelset/labels.real.jsonl \
                             --limit 60

Keys per row:
  1-9,0  pick the intent (menu shown)
  y / n  should_escalate
  s      skip this row (leaves it unlabelled)
  b      go back one row
  q      save and quit
The rubric is docs/labeling-guide.md — keep it open alongside.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from support_agent.taxonomy.intents import INTENTS

INTENT_KEYS = "1234567890"[: len(INTENTS)]
KEY_TO_INTENT = {k: it.name for k, it in zip(INTENT_KEYS, INTENTS)}


def _menu() -> str:
    return "  ".join(f"[{k}] {it.name} ({it.risk})" for k, it in zip(INTENT_KEYS, INTENTS))


def _getch() -> str:
    """Read one keypress without waiting for Enter (POSIX)."""
    import sys
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def _load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _save_jsonl(rows: list[dict], p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", default="eval/labelset/labels.real.unlabelled.jsonl")
    ap.add_argument("--out", default="eval/labelset/labels.real.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="stop after this many NEW labels")
    args = ap.parse_args()

    seed = _load_jsonl(Path(args.src))
    out_path = Path(args.out)
    done = {r["id"]: r for r in _load_jsonl(out_path) if r.get("intent")}
    todo = [r for r in seed if r["id"] not in done]

    print(f"{len(done)} already labelled · {len(todo)} remaining in seed")
    if args.limit:
        print(f"labelling up to {args.limit} this session")
    print("\nrubric: docs/labeling-guide.md\n")

    labelled: list[dict] = list(done.values())
    i = 0
    new_count = 0
    while i < len(todo):
        if args.limit and new_count >= args.limit:
            break
        row = todo[i]
        print("─" * 78)
        print(f"[{new_count + 1}{'/' + str(args.limit) if args.limit else ''}]  id={row['id']}  brand={row['brand']}")
        if row.get("history"):
            for h in row["history"]:
                print(f"   … {h}")
        print(f"\n   {row['text']}\n")
        print(_menu())
        print("[y]escalate [n]auto  ·  [s]kip [b]ack [q]uit")

        intent = None
        while intent is None:
            k = _getch().lower()
            if k == "q":
                _save_jsonl(labelled + [r for r in todo[i:] if r["id"] not in {x["id"] for x in labelled}], out_path)
                print(f"\nsaved {sum(1 for r in labelled if r.get('intent'))} labelled rows -> {out_path}")
                return
            if k == "s":
                intent = "__skip__"
            elif k == "b" and (labelled and new_count):
                labelled.pop()
                new_count -= 1
                i -= 1
                intent = "__back__"
            elif k in KEY_TO_INTENT:
                intent = KEY_TO_INTENT[k]
        if intent in ("__skip__", "__back__"):
            i += 1 if intent == "__skip__" else 0
            print(f"   -> {intent.strip('_')}")
            continue

        esc = None
        while esc is None:
            k = _getch().lower()
            if k == "y":
                esc = True
            elif k == "n":
                esc = False
        row = dict(row)
        row["intent"] = intent
        row["should_escalate"] = esc
        row["notes"] = "hand-labelled via eval.label_cli"
        labelled.append(row)
        new_count += 1
        i += 1
        print(f"   -> {intent} · escalate={esc}")

    _save_jsonl([r for r in labelled if r.get("intent")], out_path)
    print(f"\nsaved {sum(1 for r in labelled if r.get('intent'))} labelled rows -> {out_path}")


if __name__ == "__main__":
    main()
