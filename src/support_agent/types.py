"""Shared data structures passed between pipeline stages.

Deliberately plain dataclasses (not pydantic models) so they are trivial to
construct in tests and to serialise with ``dataclasses.asdict``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Turn:
    """One message in a conversation."""

    author: str  # "customer" or "agent"
    text: str


@dataclass
class Conversation:
    """A normalised support thread from the dataset."""

    conv_id: str
    brand: str
    turns: list[Turn]
    resolved: bool  # agent had the last word and the customer did not re-open

    @property
    def first_customer_text(self) -> str:
        for t in self.turns:
            if t.author == "customer":
                return t.text
        return ""

    @property
    def last_agent_text(self) -> str:
        for t in reversed(self.turns):
            if t.author == "agent":
                return t.text
        return ""


@dataclass
class CleanMessage:
    raw: str
    clean: str
    masked_spans: dict[str, int] = field(default_factory=dict)  # token -> count
    language: str = "en"
    is_english: bool = True


@dataclass
class IntentPrediction:
    intent: str
    confidence: float          # max class probability
    margin: float              # top1 - top2 probability
    distribution: dict[str, float]
    source: str = "logreg"     # or "llm_zero_shot"


@dataclass
class Exemplar:
    conv_id: str
    brand: str
    customer_text: str
    agent_text: str
    similarity: float
    same_brand: bool


@dataclass
class GeneratedReply:
    text: str
    exemplars: list[Exemplar]
    used_llm: bool
    model: str
    grounded: bool = True                 # passed the no-invented-facts heuristics
    groundedness_notes: list[str] = field(default_factory=list)


@dataclass
class Signals:
    clf_confidence: float
    clf_margin: float
    intent_risk_tier: str            # low | medium | high
    sentiment: float                 # -1..1
    anger_flag: bool
    retrieval_max_sim: float
    pii_flags: list[str] = field(default_factory=list)
    compliance_hits: list[str] = field(default_factory=list)
    is_non_english: bool = False
    history_len: int = 0


@dataclass
class RuleTrace:
    name: str
    fired: bool
    detail: str = ""


@dataclass
class EscalationDecision:
    decision: str                    # "auto_send" | "escalate"
    reason: str                      # plain-English, human-facing
    winning_rule: str
    trace: list[RuleTrace] = field(default_factory=list)


@dataclass
class Triage:
    """The full, auditable output for one inbound message."""

    brand: str
    message: CleanMessage
    intent: IntentPrediction
    reply: GeneratedReply | None
    signals: Signals
    escalation: EscalationDecision

    def summary(self) -> str:
        return (
            f"[{self.brand}] intent={self.intent.intent} "
            f"(conf={self.intent.confidence:.2f}, margin={self.intent.margin:.2f}) "
            f"-> {self.escalation.decision.upper()}: {self.escalation.reason}"
        )
