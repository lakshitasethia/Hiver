"""End-to-end triage: inbound message -> intent + reply + routing decision.

This is the object a caller (CLI, eval harness, or a future web service) uses.
It wires the components together and does nothing clever itself — all the logic
lives in the modules it calls, which keeps this file readable as a flow chart.
"""

from __future__ import annotations

from dataclasses import dataclass

from .classify.model import DEFAULT_MODEL_PATH, IntentClassifier
from .data.load import load_sample
from .data.normalize import normalize
from .embeddings import Embedder, make_embedder
from .escalate.rules import decide
from .escalate.signals import compute_signals
from .generate.generator import ReplyGenerator
from .generate.retriever import ReplyRetriever
from .llm import make_llm
from .types import Conversation, Triage


@dataclass
class SupportAgent:
    classifier: IntentClassifier
    retriever: ReplyRetriever
    generator: ReplyGenerator
    embedder: Embedder

    @classmethod
    def load(
        cls,
        *,
        model_path=DEFAULT_MODEL_PATH,
        conversations: list[Conversation] | None = None,
        embedder: Embedder | None = None,
    ) -> SupportAgent:
        emb = embedder or make_embedder()
        clf = IntentClassifier.load(model_path, embedder=emb)
        convs = conversations if conversations is not None else load_sample()
        retriever = ReplyRetriever.build(convs, embedder=emb)
        return cls(clf, retriever, ReplyGenerator(make_llm()), emb)

    def triage(
        self,
        message: str,
        brand: str,
        *,
        history: list[str] | None = None,
        draft_reply: bool = True,
    ) -> Triage:
        clean = normalize(message)
        intent = self.classifier.predict_with_confidence(clean.clean)
        exemplars = self.retriever.retrieve(clean.clean, brand)

        reply = None
        if draft_reply:
            reply = self.generator.generate(
                message, brand, exemplars, intent=intent.intent
            )

        signals = compute_signals(
            message=clean,
            intent=intent,
            exemplars=exemplars,
            history_len=len(history or []),
        )
        escalation = decide(signals, reply, intent_name=intent.intent)
        return Triage(
            brand=brand,
            message=clean,
            intent=intent,
            reply=reply,
            signals=signals,
            escalation=escalation,
        )
