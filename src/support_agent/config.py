"""Central configuration.

Every tunable number the pipeline depends on lives here so it can be found,
justified, and swept in evaluation instead of being buried as a magic literal.
Thresholds carrying a ``# tuned`` comment were selected on the dev split
(see ``eval/tune_thresholds.py``); the rest are documented defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Load a local .env (if present) before any config dataclass reads os.getenv.
try:  # pragma: no cover - trivial glue
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except Exception:
    pass

DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
CACHE_DIR = REPO_ROOT / ".cache"
RESULTS_DIR = REPO_ROOT / "eval" / "results"
SAMPLE_DATA = DATA_DIR / "sample" / "conversations.jsonl"

RANDOM_SEED = 20260909


@dataclass(frozen=True)
class EmbeddingConfig:
    # "st" = sentence-transformers (local, ~90 MB download, no key).
    # "hashing" = deterministic hashed n-grams, zero download, used in CI.
    backend: str = os.getenv("SUPPORT_AGENT_EMBEDDER", "st")
    st_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hashing_dim: int = 512
    cache: bool = True


@dataclass(frozen=True)
class LLMConfig:
    # "gemini" if GOOGLE_API_KEY is set, else "fake" (deterministic, offline).
    backend: str = os.getenv("SUPPORT_AGENT_LLM", "auto")
    model: str = os.getenv("SUPPORT_AGENT_LLM_MODEL", "gemini-flash-lite-latest")
    temperature: float = 0.3
    timeout_s: float = 30.0
    max_retries: int = 6
    # seconds to space out successive Gemini calls; raise it (e.g. 13) for a long
    # eval run so the free tier's ~5 req/min quota is never tripped.
    min_interval_s: float = float(os.getenv("SUPPORT_AGENT_LLM_MIN_INTERVAL", "0"))


@dataclass(frozen=True)
class RetrievalConfig:
    k: int = 5              # exemplars shown to the generator
    k_min: int = 2          # below this many same-brand hits, allow cross-brand
    # tuned: below this cosine sim -> escalation "weak_retrieval". Lowered
    # 0.45 -> 0.40 after both the synthetic and the real dev split independently
    # asked for a lower value (eval/tune_thresholds.py).
    weak_similarity: float = 0.40


@dataclass(frozen=True)
class EscalationConfig:
    low_confidence: float = 0.55    # tuned: max class prob below this -> escalate
    low_margin: float = 0.10        # tuned: (top1 - top2) prob below this -> escalate (0.15 -> 0.10)
    anger_sentiment: float = -0.4   # tuned: sentiment at/below this counts as "angry"
    # Escalate when the classifier put at least this much probability mass on ANY
    # high-risk intent, even if it was not the top pick.
    high_risk_mass: float = 0.25
    # Escalate a low/medium-risk intent when sentiment is at/below this — a softer
    # net than anger_sentiment, added after the real-data eval.
    routine_negative_sentiment: float = -0.2
    # Cost model for the escalation report: sending a bad auto-reply is much
    # worse than needlessly escalating a message a human then rubber-stamps.
    cost_false_auto_send: float = 5.0
    cost_false_escalate: float = 1.0


@dataclass(frozen=True)
class Config:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    escalation: EscalationConfig = field(default_factory=EscalationConfig)
    max_history_turns: int = 4

    def with_overrides(self, **kw) -> Config:
        return replace(self, **kw)


CONFIG = Config()


def ensure_dirs() -> None:
    for d in (DATA_DIR, MODELS_DIR, CACHE_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
