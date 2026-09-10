from __future__ import annotations

import os
import re
import time

from .base import LLMResult

_RETRY_AFTER = re.compile(r"(?:retry[- ]after|try again in)\D*(\d+(?:\.\d+)?)\s*(m|ms|s)?", re.I)

# Groq's chat models change often; these are the current text models as of the
# last check. gpt-oss-20b is fast and generous on limits; qwen is a different
# family, handy as the LLM-judge to reduce "model grades its own output" bias.
DEFAULT_MODEL = "openai/gpt-oss-20b"


class GroqClient:
    """Groq API (OpenAI-shaped) via the official ``groq`` SDK.

    Free tier is far more generous than Gemini's (~1000 req/day, ~8k tokens/min
    per model at time of writing), which is what makes a full evaluation run
    feasible without paying. Reads ``GROQ_API_KEY``. Honours the ``retry-after``
    hint on 429s and supports the same ``min_interval_s`` throttle as the Gemini
    client so a long run can be paced under the tokens-per-minute cap.
    """

    _last_call_at: float = 0.0

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        timeout_s: float = 120.0,
        max_retries: int = 6,
        min_interval_s: float | None = None,
    ) -> None:
        import groq  # imported lazily so the no-key path never needs it

        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys "
                "or run with SUPPORT_AGENT_LLM=fake."
            )
        self._groq = groq
        self._client = groq.Groq(api_key=key, timeout=timeout_s, max_retries=0)
        self.model = model or os.getenv("SUPPORT_AGENT_GROQ_MODEL") or DEFAULT_MODEL
        self.name = f"groq:{self.model}"
        self.max_retries = max_retries
        self.min_interval_s = (
            float(os.getenv("SUPPORT_AGENT_LLM_MIN_INTERVAL", "0"))
            if min_interval_s is None
            else min_interval_s
        )

    def _throttle(self) -> None:
        if self.min_interval_s <= 0:
            return
        wait = self.min_interval_s - (time.monotonic() - GroqClient._last_call_at)
        if wait > 0:
            time.sleep(wait)
        GroqClient._last_call_at = time.monotonic()

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.3,
        json_mode: bool = False,
    ) -> LLMResult:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1024,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        # gpt-oss models reason before answering; keep that budget small — this is
        # short-form drafting/scoring, not a puzzle — to save latency and tokens.
        if "gpt-oss" in self.model:
            kwargs["reasoning_effort"] = "low"

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                resp = self._client.chat.completions.create(**kwargs)
                msg = resp.choices[0].message
                usage = getattr(resp, "usage", None)
                return LLMResult(
                    text=(msg.content or "").strip(),
                    model=self.name,
                    prompt_tokens=getattr(usage, "prompt_tokens", None),
                    completion_tokens=getattr(usage, "completion_tokens", None),
                )
            except Exception as exc:  # noqa: BLE001 - backoff on anything transient
                last_err = exc
                if not _is_retryable(exc) or attempt == self.max_retries - 1:
                    raise
                time.sleep(_retry_delay(exc, default=2**attempt))
        raise last_err  # pragma: no cover


def _is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status in (408, 409, 425, 429, 500, 502, 503, 504):
        return True
    # groq SDK exception class names: APITimeoutError, APIConnectionError,
    # InternalServerError, RateLimitError
    if type(exc).__name__ in (
        "APITimeoutError", "APIConnectionError", "InternalServerError", "RateLimitError"
    ):
        return True
    text = f"{exc}".lower()
    return any(
        s in text
        for s in ("429", "rate limit", "overloaded", "timeout", "timed out",
                  "connection", "503", "502", "try again")
    )


def _retry_delay(exc: Exception, *, default: float) -> float:
    m = _RETRY_AFTER.search(str(exc))
    if not m:
        return default
    val, unit = float(m.group(1)), (m.group(2) or "s").lower()
    seconds = val / 1000 if unit == "ms" else val * 60 if unit == "m" else val
    return min(90.0, seconds + 1.0)
