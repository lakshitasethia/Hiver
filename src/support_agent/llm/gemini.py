from __future__ import annotations

import os
import time

from .base import LLMResult


class GeminiClient:
    """Google AI Studio (Gemini) via the ``google-genai`` SDK.

    Free tier is enough for this project. Reads ``GOOGLE_API_KEY``. Retries on
    transient errors (429 / 5xx) with exponential backoff, and enforces a hard
    per-call timeout so a hung request cannot stall an eval run.
    """

    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        *,
        api_key: str | None = None,
        timeout_s: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        from google import genai  # imported lazily so the no-key path never needs it

        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Get a free key at "
                "https://aistudio.google.com/apikey or run with SUPPORT_AGENT_LLM=fake."
            )
        self._genai = genai
        self._client = genai.Client(api_key=key)
        self.model = model
        self.name = f"gemini:{model}"
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.3,
        json_mode: bool = False,
    ) -> LLMResult:
        from google.genai import types

        cfg = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system,
            response_mime_type="application/json" if json_mode else None,
            http_options=types.HttpOptions(timeout=int(self.timeout_s * 1000)),
        )
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.models.generate_content(
                    model=self.model, contents=prompt, config=cfg
                )
                usage = getattr(resp, "usage_metadata", None)
                return LLMResult(
                    text=(resp.text or "").strip(),
                    model=self.name,
                    prompt_tokens=getattr(usage, "prompt_token_count", None),
                    completion_tokens=getattr(usage, "candidates_token_count", None),
                )
            except Exception as exc:  # noqa: BLE001 - backoff on anything transient
                last_err = exc
                if not _is_retryable(exc) or attempt == self.max_retries - 1:
                    raise
                time.sleep(2**attempt)
        raise last_err  # pragma: no cover


def _is_retryable(exc: Exception) -> bool:
    text = f"{exc}".lower()
    return any(s in text for s in ("429", "resource_exhausted", "500", "503", "unavailable", "timeout"))
