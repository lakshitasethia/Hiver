import pytest

from support_agent.classify.weak_label import coverage, weak_label

CASES = [
    ("I was charged twice for order 12345678, refund the duplicate now", "billing_dispute"),
    ("Please cancel my subscription before it renews", "cancellation"),
    ("I can't log in and the password reset email never arrives", "account_access"),
    ("My parcel arrived damaged, the box was crushed", "delivery_issue"),
    ("Where is my order? Tracking hasn't moved in days", "order_status"),
    ("The app crashes every time I open the payments tab", "technical_bug"),
    ("Does the plan include international roaming?", "product_question"),
    ("Thank you so much, your agent was fantastic", "praise_thanks"),
    ("hi", "other_unclear"),
]


@pytest.mark.parametrize("text,expected", CASES)
def test_weak_label_hits_expected(text, expected):
    intent, conf, _fired = weak_label(text)
    assert intent == expected
    assert conf > 0.5


def test_weak_label_abstains_on_ambiguous():
    intent, conf, fired = weak_label("I have a question about the thing from yesterday")
    assert intent is None
    assert conf == 0.0
    assert fired == []


def test_priority_beats_count():
    # generic 'refund' cue + strong 'cancel my account' cue -> cancellation wins
    intent, _c, _f = weak_label("I want a refund and to cancel my account entirely")
    assert intent == "cancellation"


def test_coverage_shape(corpus):
    texts = [c.first_customer_text for c in corpus]
    cov = coverage(texts)
    assert cov["n"] == len(texts)
    assert 0.0 <= cov["coverage"] <= 1.0
    assert set(cov["per_intent"]) == {
        "order_status", "delivery_issue", "billing_dispute", "cancellation",
        "account_access", "technical_bug", "product_question",
        "complaint_feedback", "praise_thanks", "other_unclear",
    }
