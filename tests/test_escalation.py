from support_agent.data.normalize import normalize
from support_agent.escalate.rules import confidence_only_baseline, decide
from support_agent.escalate.signals import compute_signals, sentiment_score
from support_agent.types import Exemplar, IntentPrediction, Signals


def _sig(**kw) -> Signals:
    base = dict(
        clf_confidence=0.9, clf_margin=0.4, intent_risk_tier="low", sentiment=0.0,
        anger_flag=False, retrieval_max_sim=0.8, pii_flags=[], compliance_hits=[],
        severity_hits=[], high_risk_mass=0.0, is_non_english=False, history_len=0,
    )
    base.update(kw)
    return Signals(**base)


def test_clean_message_auto_sends():
    d = decide(_sig())
    assert d.decision == "auto_send"
    assert d.winning_rule == "none"


def test_compliance_keyword_forces_escalation():
    d = decide(_sig(compliance_hits=["chargeback"]))
    assert d.decision == "escalate"
    assert d.winning_rule == "compliance_or_legal"
    assert "chargeback" in d.reason


def test_high_risk_intent_escalates_even_when_confident():
    d = decide(_sig(intent_risk_tier="high"), intent_name="cancellation")
    assert d.decision == "escalate"
    assert d.winning_rule == "high_risk_intent"


def test_low_confidence_escalates():
    d = decide(_sig(clf_confidence=0.4))
    assert d.decision == "escalate"
    assert d.winning_rule == "low_model_confidence"


def test_low_margin_escalates():
    d = decide(_sig(clf_margin=0.05))
    assert d.decision == "escalate"


def test_weak_retrieval_escalates():
    d = decide(_sig(retrieval_max_sim=0.2))
    assert d.decision == "escalate"
    assert d.winning_rule == "weak_retrieval"


def test_rule_order_compliance_before_confidence():
    # both would fire; compliance must win because it is listed first
    d = decide(_sig(clf_confidence=0.2, compliance_hits=["gdpr"]))
    assert d.winning_rule == "compliance_or_legal"


def test_severity_cue_escalates_regardless_of_intent():
    # low-risk intent, confident, calm sentiment — only the severity cue fires
    d = decide(_sig(severity_hits=["theft_loss"]))
    assert d.decision == "escalate"
    assert d.winning_rule == "severity_cue"
    assert "theft" in d.reason


def test_high_risk_mass_escalates_when_not_top_pick():
    d = decide(_sig(intent_risk_tier="low", high_risk_mass=0.3))
    assert d.decision == "escalate"
    assert d.winning_rule == "high_risk_intent_suspected"


def test_negative_sentiment_on_routine_intent_escalates():
    d = decide(_sig(intent_risk_tier="medium", sentiment=-0.35))
    assert d.decision == "escalate"
    assert d.winning_rule == "negative_sentiment_routine_intent"


def test_trace_is_complete():
    d = decide(_sig(intent_risk_tier="high"))
    names = {t.name for t in d.trace}
    assert {"non_english", "severity_cue", "high_risk_intent_suspected",
            "negative_sentiment_routine_intent", "weak_retrieval"} <= names
    assert len(d.trace) == len({t.name for t in d.trace})  # names unique
    assert len(d.trace) == 11


def test_confidence_only_baseline():
    assert confidence_only_baseline(_sig(clf_confidence=0.4), 0.55).decision == "escalate"
    assert confidence_only_baseline(_sig(clf_confidence=0.9), 0.55).decision == "auto_send"


def test_sentiment_direction():
    assert sentiment_score("this is absolutely terrible and unacceptable, worst ever!!") < -0.2
    assert sentiment_score("thank you so much, this was excellent and perfect") > 0.2


def test_compute_signals_end_to_end():
    clean = normalize("If you don't refund me I'm filing a chargeback. Order #AB12345.")
    intent = IntentPrediction("billing_dispute", 0.7, 0.3, {"billing_dispute": 0.7})
    ex = [Exemplar("c1", "Acme", "q", "a", 0.6, True)]
    s = compute_signals(message=clean, intent=intent, exemplars=ex)
    assert "chargeback" in s.compliance_hits
    assert "order_id" in s.pii_flags
    assert s.intent_risk_tier == "high"
    assert s.high_risk_mass == 0.7  # all mass on a high-risk intent


def test_compute_signals_severity_from_raw_text():
    clean = normalize("you all stole my package and it never arrived")
    intent = IntentPrediction("delivery_issue", 0.8, 0.6, {"delivery_issue": 0.8})
    s = compute_signals(message=clean, intent=intent, exemplars=[])
    assert "theft_loss" in s.severity_hits
