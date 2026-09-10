"""Swappable LLM client.

Everything downstream depends only on the ``LLMClient`` protocol, so the backend
(Groq, Gemini, a deterministic fake for tests/CI, or a future provider) can be
swapped without touching the pipeline.
"""

from .base import LLMClient, LLMResult
from .factory import make_judge_llm, make_llm
from .fake import FakeLLM

__all__ = ["LLMClient", "LLMResult", "FakeLLM", "make_llm", "make_judge_llm"]
