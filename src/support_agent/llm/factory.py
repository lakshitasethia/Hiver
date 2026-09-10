from __future__ import annotations

import os

from ..config import CONFIG
from .base import LLMClient
from .fake import FakeLLM

_GEMINI_KEYS = ("GOOGLE_API_KEY", "GEMINI_API_KEY")


def _has(*names: str) -> bool:
    return any(os.getenv(n) for n in names)


def make_llm(backend: str | None = None, *, model: str | None = None) -> LLMClient:
    """Pick an LLM client.

    ``auto`` (default) resolves in this order: Groq if ``GROQ_API_KEY`` is set,
    then Gemini if ``GOOGLE_API_KEY`` is set, otherwise the deterministic offline
    fake. ``groq`` / ``gemini`` / ``fake`` force the choice. ``model`` overrides
    the configured model id for this client only (used to run the LLM-judge on a
    different model family than the generator).
    """
    backend = backend or CONFIG.llm.backend

    if backend == "fake":
        return FakeLLM(model=model or CONFIG.llm.model)

    if backend == "auto":
        backend = "groq" if _has("GROQ_API_KEY") else "gemini" if _has(*_GEMINI_KEYS) else "fake"
        if backend == "fake":
            return FakeLLM(model=CONFIG.llm.model)

    if backend == "groq":
        from .groq import GroqClient

        return GroqClient(
            model=model,  # None -> GroqClient's own default
            timeout_s=CONFIG.llm.timeout_s,
            max_retries=CONFIG.llm.max_retries,
            min_interval_s=CONFIG.llm.min_interval_s,
        )

    if backend == "gemini":
        if not _has(*_GEMINI_KEYS):
            raise RuntimeError("SUPPORT_AGENT_LLM=gemini but GOOGLE_API_KEY is unset.")
        from .gemini import GeminiClient

        m = model or CONFIG.llm.model
        return GeminiClient(
            model=m,
            timeout_s=CONFIG.llm.timeout_s,
            max_retries=CONFIG.llm.max_retries,
            min_interval_s=CONFIG.llm.min_interval_s,
        )

    raise ValueError(f"unknown LLM backend: {backend!r}")


def make_judge_llm() -> LLMClient:
    """The client for the LLM-as-judge.

    Defaults to the same backend as the generator but a different model when
    ``SUPPORT_AGENT_JUDGE_MODEL`` is set, so "the judge grades its own family" is
    at least mitigated. Falls back to ``make_llm()`` when unset.
    """
    judge_model = os.getenv("SUPPORT_AGENT_JUDGE_MODEL")
    if not judge_model:
        return make_llm()
    backend = os.getenv("SUPPORT_AGENT_JUDGE_BACKEND") or CONFIG.llm.backend
    return make_llm(backend if backend != "auto" else None, model=judge_model)
