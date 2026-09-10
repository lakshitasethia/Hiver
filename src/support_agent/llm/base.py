from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class LLMResult:
    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    raw: dict = field(default_factory=dict)


@runtime_checkable
class LLMClient(Protocol):
    """Minimal surface the pipeline needs from any language model."""

    name: str

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.3,
        json_mode: bool = False,
    ) -> LLMResult:
        ...
