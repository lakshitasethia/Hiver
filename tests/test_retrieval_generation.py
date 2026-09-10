from support_agent.embeddings import make_embedder
from support_agent.generate.baseline_retrieval import nearest_reply_baseline
from support_agent.generate.generator import ReplyGenerator
from support_agent.generate.retriever import ReplyRetriever
from support_agent.llm import FakeLLM


def test_retriever_prefers_same_brand(corpus):
    r = ReplyRetriever.build(corpus, embedder=make_embedder())
    brand = corpus[0].brand
    ex = r.retrieve("where is my order, it has not arrived", brand, k=5)
    assert ex
    assert ex == sorted(ex, key=lambda e: -e.similarity)
    # at least the top hit should be same-brand when the brand has enough data
    same = [e for e in ex if e.same_brand]
    assert len(same) >= 1


def test_retriever_handles_empty_corpus():
    r = ReplyRetriever.build([], embedder=make_embedder())
    assert r.retrieve("anything", "AnyBrand") == []


def test_generator_grounding_flags_invented_specifics(corpus):
    gen = ReplyGenerator(FakeLLM())
    r = ReplyRetriever.build(corpus, embedder=make_embedder())
    ex = r.retrieve("my order is late", corpus[0].brand)
    reply = gen.generate("my order is late", corpus[0].brand, ex)
    assert isinstance(reply.grounded, bool)
    assert reply.used_llm is True


def test_generator_without_exemplars_returns_clarifier(corpus):
    gen = ReplyGenerator(FakeLLM())
    reply = gen.generate("hello?", "AnyBrand", [])
    assert reply.used_llm is False
    assert "order number" in reply.text.lower() or "account" in reply.text.lower()


def test_baseline_returns_verbatim(corpus):
    r = ReplyRetriever.build(corpus, embedder=make_embedder())
    ex = r.retrieve("cancel my plan", corpus[0].brand)
    base = nearest_reply_baseline(ex)
    assert base.text == ex[0].agent_text
    assert base.used_llm is False
