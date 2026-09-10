"""Load and thread the raw dataset into :class:`Conversation` objects.

Source: Kaggle "Customer Support on Twitter"
(``thoughtvector/customer-support-on-twitter``), file ``twcs/twcs.csv`` with
columns ``tweet_id, author_id, inbound, created_at, text, response_tweet_id,
in_response_to_tweet_id``.

Threading: follow ``in_response_to_tweet_id`` back to a root, then walk forward
along ``response_tweet_id``. A turn's author is ``customer`` when ``inbound`` is
true, otherwise ``agent``; the brand is the first agent author in the thread.
``resolved`` is a *proxy*: the last turn is the agent's and no customer turn
follows it (documented as a limitation in the report).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from ..config import SAMPLE_DATA
from ..types import Conversation, Turn


def _walk_forward(root_id: str, by_id: dict, children: dict) -> list:
    seen: set[str] = set()
    order: list = []
    frontier = [root_id]
    while frontier:
        tid = frontier.pop(0)
        if tid in seen or tid not in by_id:
            continue
        seen.add(tid)
        order.append(by_id[tid])
        nxt = children.get(tid, [])
        frontier.extend(sorted(nxt, key=lambda x: by_id[x]["created_at"] if x in by_id else ""))
    order.sort(key=lambda r: r["created_at"])
    return order


def load_conversations(
    csv_path: str | Path,
    *,
    max_threads: int | None = None,
    min_turns: int = 2,
) -> list[Conversation]:
    import pandas as pd

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    df["inbound"] = df["inbound"].astype(str).str.lower().eq("true")
    by_id = {r["tweet_id"]: r for r in df.to_dict("records")}

    children: dict[str, list[str]] = {}
    has_parent: set[str] = set()
    for r in by_id.values():
        parent = r["in_response_to_tweet_id"]
        if parent:
            children.setdefault(parent, []).append(r["tweet_id"])
            has_parent.add(r["tweet_id"])

    roots = [
        tid
        for tid, r in by_id.items()
        if r["inbound"] and tid not in has_parent
    ]
    roots.sort(key=lambda t: by_id[t]["created_at"])

    out: list[Conversation] = []
    for root in roots:
        rows = _walk_forward(root, by_id, children)
        if len(rows) < min_turns:
            continue
        brand = next((r["author_id"] for r in rows if not r["inbound"]), "")
        if not brand:
            continue
        turns = [
            Turn(author="customer" if r["inbound"] else "agent", text=r["text"])
            for r in rows
            if r["text"].strip()
        ]
        if len(turns) < min_turns:
            continue
        resolved = turns[-1].author == "agent"
        out.append(
            Conversation(conv_id=root, brand=brand, turns=turns, resolved=resolved)
        )
        if max_threads and len(out) >= max_threads:
            break
    return out


def _conv_from_dict(d: dict) -> Conversation:
    return Conversation(
        conv_id=d["conv_id"],
        brand=d["brand"],
        turns=[Turn(**t) for t in d["turns"]],
        resolved=bool(d["resolved"]),
    )


def load_jsonl(path: str | Path) -> list[Conversation]:
    lines = Path(path).read_text().splitlines()
    return [_conv_from_dict(json.loads(ln)) for ln in lines if ln.strip()]


def load_sample() -> list[Conversation]:
    """The committed, credential-free fixture (see ``data/sample/``)."""
    if not SAMPLE_DATA.exists():
        raise FileNotFoundError(
            f"{SAMPLE_DATA} missing. Run `make data-sample` to generate it."
        )
    return load_jsonl(SAMPLE_DATA)


def dump_jsonl(conversations: Iterable[Conversation], path: str | Path) -> None:
    from dataclasses import asdict

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for c in conversations:
            fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def iter_first_customer_messages(
    conversations: Iterable[Conversation],
) -> Iterator[tuple[str, str, str]]:
    """Yield ``(conv_id, brand, first_customer_text)`` for classifier work."""
    for c in conversations:
        text = c.first_customer_text
        if text:
            yield c.conv_id, c.brand, text
