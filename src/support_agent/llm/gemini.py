from __future__ import annotations

import os
import re
import time

from .base import LLMResult

_RETRY_AFTER = re.compile(r"retry(?:delay|.{0,12}?in)\D*(\d+(?:\.\d+)?)\s*s", re.I)


class GeminiClient:
    """Google AI Studio (Gemini) via the ``google-genai`` SDK.

    Reads ``GOOGLE_API_KEY``. On a transient error (429 / 5xx) it retries, and
    when the response carries a ``retryDelay`` (the free tier's per-minute quota
    does) it waits exactly that long instead of guessing. An optional
    ``min_interval_s`` throttle spaces successive calls out — set it via
    ``SUPPORT_AGENT_LLM_MIN_INTERVAL`` so a long eval run stays under the
    free-tier requests-per-minute cap without tripping 429s at all.
    """

    # class-level so the throttle spans every client instance in a process
    _last_call_at: float = 0.0

    def __init__(
        self,
        model: str = "gemini-flash-lite-latest",
        *,
        api_key: str | None = None,
        timeout_s: float = 30.0,
        max_retries: int = 6,
        min_interval_s: float | None = None,
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
        self.min_interval_s = (
            float(os.getenv("SUPPORT_AGENT_LLM_MIN_INTERVAL", "0"))
            if min_interval_s is None
            else min_interval_s
        )

    def _throttle(self) -> None:
        if self.min_interval_s <= 0:
            return
        wait = self.min_interval_s - (time.monotonic() - GeminiClient._last_call_at)
        if wait > 0:
            time.sleep(wait)
        GeminiClient._last_call_at = time.monotonic()

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
            self._throttle()
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
                time.sleep(_retry_delay(exc, default=2**attempt))
        raise last_err  # pragma: no cover


def _is_retryable(exc: Exception) -> bool:
    text = f"{exc}".lower()
    return any(
        s in text
        for s in ("429", "resource_exhausted", "500", "502", "503", "unavailable", "timeout", "deadline")
    )


def _retry_delay(exc: Exception, *, default: float) -> float:
    m = _RETRY_AFTER.search(str(exc))
    if m:
        return min(90.0, float(m.group(1)) + 1.0)  # honour server hint, cap the wait
    return default
