import json

from support_agent.llm import FakeLLM, make_llm
from support_agent.llm.factory import make_llm as factory_make


def test_fake_llm_classify_dispatch():
    f = FakeLLM()
    out = f.complete('Return the single best intent label as JSON. Customer: "cancel my account"')
    assert json.loads(out.text)["intent"] == "cancellation"


def test_fake_llm_generate_dispatch():
    f = FakeLLM()
    out = f.complete('Brand: Acme\nExample 1\nAgent: Here is your tracking link.\n\nNEW message:\nCustomer: where is it?\n\nYour reply:')
    assert len(out.text) > 20
    assert "tracking" in out.text.lower()


def test_fake_llm_judge_dispatch():
    f = FakeLLM()
    prompt = 'Rate on helpfulness, tone_match, factual_caution.\nCandidate reply:\n"""Thanks, your refund of has been processed and will arrive soon enough."""'
    out = f.complete(prompt)
    d = json.loads(out.text)
    assert set(d) >= {"helpfulness", "tone_match", "factual_caution", "would_send"}


def test_factory_returns_fake_without_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert "fake" in factory_make("auto").name


def test_factory_auto_prefers_groq(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    import sys
    import types as _t

    stub = _t.ModuleType("groq")
    stub.Groq = lambda **kw: object()
    monkeypatch.setitem(sys.modules, "groq", stub)
    assert factory_make("auto").name.startswith("groq:")


def test_groq_client_shapes_request(monkeypatch):
    """GroqClient builds an OpenAI-style call and parses the response."""
    import sys
    import types as _t

    seen = {}

    class _Resp:
        def __init__(self):
            self.choices = [_t.SimpleNamespace(message=_t.SimpleNamespace(content=' {"intent": "cancellation"} '))]
            self.usage = _t.SimpleNamespace(prompt_tokens=11, completion_tokens=7)

    class _Client:
        def __init__(self, **kw):
            self.chat = _t.SimpleNamespace(
                completions=_t.SimpleNamespace(create=self._create)
            )

        def _create(self, **kw):
            seen.update(kw)
            return _Resp()

    stub = _t.ModuleType("groq")
    stub.Groq = _Client
    monkeypatch.setitem(sys.modules, "groq", stub)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    from support_agent.llm.groq import GroqClient

    c = GroqClient(model="openai/gpt-oss-20b")
    out = c.complete("classify this", system="be terse", temperature=0.0, json_mode=True)
    assert out.text == '{"intent": "cancellation"}'
    assert out.prompt_tokens == 11 and out.completion_tokens == 7
    assert seen["response_format"] == {"type": "json_object"}
    assert seen["reasoning_effort"] == "low"  # gpt-oss path
    assert seen["messages"][0] == {"role": "system", "content": "be terse"}


def test_pipeline_triage_smoke(corpus, tmp_path):
    from support_agent.classify.model import train_classifier
    from support_agent.classify.weak_label import weak_label
    from support_agent.embeddings import make_embedder
    from support_agent.generate.generator import ReplyGenerator
    from support_agent.generate.retriever import ReplyRetriever
    from support_agent.pipeline import SupportAgent

    emb = make_embedder()
    texts, labels = [], []
    for c in corpus:
        wl, conf, _ = weak_label(c.first_customer_text)
        if wl and conf >= 0.6:
            texts.append(c.first_customer_text)
            labels.append(wl)
    clf = train_classifier(texts, labels, embedder=emb)

    agent = SupportAgent(
        classifier=clf,
        retriever=ReplyRetriever.build(corpus, embedder=emb),
        generator=ReplyGenerator(make_llm()),
        embedder=emb,
    )
    t = agent.triage("If you don't refund me today I'm calling my lawyer", corpus[0].brand)
    assert t.escalation.decision == "escalate"
    assert "legal" in t.escalation.reason.lower() or "lawyer" in t.escalation.reason.lower()
    assert t.reply is not None
    assert t.summary()
