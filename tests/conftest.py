"""Test config: force the deterministic, offline backends for every test."""

import os

os.environ.setdefault("SUPPORT_AGENT_EMBEDDER", "hashing")
os.environ.setdefault("SUPPORT_AGENT_LLM", "fake")

import pytest

from support_agent.data.synthetic import build_corpus


@pytest.fixture(scope="session")
def corpus():
    return [c for c, _intent in build_corpus(n=120, seed=1)]


@pytest.fixture(scope="session")
def labelled_corpus():
    return build_corpus(n=120, seed=1)
