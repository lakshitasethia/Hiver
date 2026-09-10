"""Swappable LLM client.

Everything downstream depends only on the ``LLMClient`` protocol, so the Gemini
backend can be replaced with a fake (tests, offline demo) or another provider
without touching the pipeline.
"""

from .base import LLMClient, LLMResult
from .factory import make_llm
from .fake import FakeLLM

__all__ = ["LLMClient", "LLMResult", "FakeLLM", "make_llm"]
