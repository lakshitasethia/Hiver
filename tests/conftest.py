"""Test config: force the deterministic, offline backends for every test."""

import os

os.environ.setdefault("SUPPORT_AGENT_EMBEDDER", "hashing")
os.environ.setdefault("SUPPORT_AGENT_LLM", "fake")
# Keep the suite hermetic even when a developer has a real key in .env:
# the fake backend must be the only one the factory can reach.
for _k in ("GROQ_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "SUPPORT_AGENT_JUDGE_MODEL"):
    os.environ.pop(_k, None)

import pytest  # noqa: E402  (env must be set before support_agent is imported)

from support_agent.data.synthetic import build_corpus  # noqa: E402


@pytest.fixture(scope="session")
def corpus():
    return [c for c, _intent in build_corpus(n=120, seed=1)]


@pytest.fixture(scope="session")
def labelled_corpus():
    return build_corpus(n=120, seed=1)
