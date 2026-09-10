from __future__ import annotations

import os

from ..config import CONFIG
from .base import LLMClient
from .fake import FakeLLM


def make_llm(backend: str | None = None) -> LLMClient:
    """Pick an LLM client.

    ``auto`` (the default): Gemini when ``GOOGLE_API_KEY`` is present, otherwise
    the deterministic fake. Explicit ``gemini`` / ``fake`` force the choice.
    """
    backend = backend or CONFIG.llm.backend
    if backend == "fake":
        return FakeLLM(model=CONFIG.llm.model)
    if backend in ("auto", "gemini"):
        if os.getenv("GOOGLE_API_KEY"):
            from .gemini import GeminiClient

            return GeminiClient(
                model=CONFIG.llm.model,
                timeout_s=CONFIG.llm.timeout_s,
                max_retries=CONFIG.llm.max_retries,
            )
        if backend == "gemini":
            raise RuntimeError("SUPPORT_AGENT_LLM=gemini but GOOGLE_API_KEY is unset.")
        return FakeLLM(model=CONFIG.llm.model)
    raise ValueError(f"unknown LLM backend: {backend!r}")
