"""Deterministic text normalisation.

Support tweets are noisy: @handles, shortened URLs, boilerplate signatures,
zero-width characters. We clean them into a stable form for embedding while
keeping the raw text and a record of what was masked, so nothing disappears
silently and the escalation layer can still see, e.g., that an email address
was present.
"""

from __future__ import annotations

import re

from ..types import CleanMessage

_MENTION = re.compile(r"@\w{1,20}")
_URL = re.compile(r"https?://\S+|\bwww\.\S+")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?<!\w)(\+?\d[\d\s().-]{7,}\d)(?!\w)")
_ORDER = re.compile(r"\b(?:order|ref|case|tracking)\s*#?\s*([A-Z0-9]{5,})\b", re.I)
_SIGNATURE = re.compile(r"(?im)^\s*(?:-{1,2}\s*\w+|sent from my \w+|regards,?|thanks,?\s*\w+)\s*$")
_WS = re.compile(r"\s+")
_ZERO_WIDTH = re.compile("[​‌‍‎‏﻿­]")

# Small English stopword set for a cheap language gate. If too few of these show
# up relative to token count, the message probably is not English and we escalate
# rather than feed a bad embedding into the classifier.
_EN_STOP = {
    "the", "a", "an", "and", "or", "to", "of", "in", "is", "it", "you", "i",
    "my", "me", "for", "on", "with", "this", "that", "have", "not", "was",
    "we", "your", "at", "be", "are", "can", "do", "no", "so", "but",
}


def _mask(text: str, pattern: re.Pattern, token: str, counts: dict[str, int]) -> str:
    def repl(_m: re.Match) -> str:
        counts[token] = counts.get(token, 0) + 1
        return token

    return pattern.sub(repl, text)


def detect_english(clean: str) -> bool:
    tokens = re.findall(r"[a-zA-Z']+", clean.lower())
    if len(tokens) < 4:
        return True  # too short to judge; let downstream signals handle it
    hits = sum(1 for t in tokens if t in _EN_STOP)
    return hits / len(tokens) >= 0.12


def normalize(text: str) -> CleanMessage:
    raw = text or ""
    counts: dict[str, int] = {}
    s = _ZERO_WIDTH.sub("", raw)
    s = _SIGNATURE.sub("", s)
    s = _mask(s, _EMAIL, "<EMAIL>", counts)
    s = _mask(s, _URL, "<URL>", counts)
    s = _mask(s, _ORDER, "<ORDERID>", counts)
    s = _mask(s, _PHONE, "<PHONE>", counts)
    s = _mask(s, _MENTION, "<USER>", counts)
    s = _WS.sub(" ", s).strip()
    is_en = detect_english(s)
    return CleanMessage(
        raw=raw,
        clean=s,
        masked_spans=counts,
        language="en" if is_en else "non-en",
        is_english=is_en,
    )
