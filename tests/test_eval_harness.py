"""The eval harness must run end-to-end on a tiny slice with the offline backends."""

from eval.common import LabelRow, binary_report, multiclass_report
from support_agent.classify.model import train_classifier
from support_agent.classify.weak_label import weak_label
from support_agent.embeddings import make_embedder


def test_metric_primitives():
    r = binary_report([True, True, False, False], [True, False, False, False])
    assert r["tp"] == 1 and r["fn"] == 1 and r["tn"] == 2
    assert r["precision"] == 1.0

    m = multiclass_report(["a", "b", "a"], ["a", "b", "b"], ["a", "b"])
    assert m["accuracy"] == round(2 / 3, 4)
    assert m["confusion_matrix"]["a"]["b"] == 1


def test_classification_eval_runs(labelled_corpus):
    # Train an in-memory classifier and pass it in, so the test never touches
    # the committed models/clf.joblib.
    emb = make_embedder()
    texts, labels = [], []
    for conv, _i in labelled_corpus:
        wl, conf, _ = weak_label(conv.first_customer_text)
        if wl and conf >= 0.6:
            texts.append(conv.first_customer_text)
            labels.append(wl)
    clf = train_classifier(texts, labels, embedder=emb)

    rows = [
        LabelRow(id=c.conv_id, brand=c.brand, text=c.first_customer_text,
                 intent=i, should_escalate=False)
        for c, i in labelled_corpus[:20]
    ]
    from eval import run_classification

    out = run_classification.evaluate(rows, run_llm_baseline=False, classifier=clf)
    assert 0.0 <= out["primary"]["macro_f1"] <= 1.0
    assert out["primary"]["n"] == 20


def test_escalation_eval_runs(labelled_corpus, corpus):
    from eval import run_escalation
    from support_agent.classify.model import train_classifier
    from support_agent.generate.retriever import ReplyRetriever

    emb = make_embedder()
    texts, labels = [], []
    for conv, _i in labelled_corpus:
        wl, conf, _ = weak_label(conv.first_customer_text)
        if wl and conf >= 0.6:
            texts.append(conv.first_customer_text)
            labels.append(wl)
    clf = train_classifier(texts, labels, embedder=emb)
    retr = ReplyRetriever.build(corpus, embedder=emb)

    rows = [
        LabelRow(id=c.conv_id, brand=c.brand, text=c.first_customer_text,
                 intent=i, should_escalate=(i in ("billing_dispute", "cancellation", "account_access")))
        for c, i in labelled_corpus[:25]
    ]
    out = run_escalation.evaluate(rows, classifier=clf, retriever=retr)
    assert out["primary_rule_engine"]["n"] == 25
    assert "weighted_cost" in out["primary_rule_engine"]
    assert out["sweep"]
