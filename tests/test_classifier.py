import numpy as np
import pytest

from support_agent.classify.model import train_classifier
from support_agent.classify.weak_label import weak_label
from support_agent.embeddings import make_embedder


@pytest.fixture(scope="module")
def trained(labelled_corpus):
    emb = make_embedder()
    texts, labels = [], []
    for conv, _intent in labelled_corpus:
        t = conv.first_customer_text
        wl, conf, _ = weak_label(t)
        if wl and conf >= 0.6:
            texts.append(t)
            labels.append(wl)
    return train_classifier(texts, labels, embedder=emb)


def test_predict_shape_and_bounds(trained):
    pred = trained.predict_with_confidence("where is my order it hasn't arrived")
    assert pred.intent in trained.classes
    assert 0.0 <= pred.confidence <= 1.0
    assert -1.0 <= pred.margin <= 1.0
    assert abs(sum(pred.distribution.values()) - 1.0) < 1e-6


def test_batch_matches_single(trained):
    texts = ["cancel my subscription now", "the app keeps crashing on checkout"]
    batch = trained.predict_batch(texts)
    for t, b in zip(texts, batch):
        assert trained.predict_with_confidence(t).intent == b.intent


def test_report_records_provenance(trained):
    r = trained.report
    assert r.n_train > 0
    assert r.n_classes >= 2
    assert len(r.data_sha1) == 40


def test_roundtrip_save_load(trained, tmp_path):
    p = tmp_path / "clf.joblib"
    trained.save(p)
    from support_agent.classify.model import IntentClassifier

    loaded = IntentClassifier.load(p, embedder=make_embedder())
    a = trained.predict_with_confidence("I need a refund for a double charge")
    b = loaded.predict_with_confidence("I need a refund for a double charge")
    assert a.intent == b.intent
    assert np.isclose(a.confidence, b.confidence)
