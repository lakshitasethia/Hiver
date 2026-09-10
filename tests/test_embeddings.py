import numpy as np

from support_agent.embeddings import HashingEmbedder, make_embedder


def test_hashing_embedder_deterministic():
    e = HashingEmbedder(dim=256)
    a = e.encode(["where is my order"])
    b = e.encode(["where is my order"])
    assert np.allclose(a, b)
    assert a.shape == (1, 256)


def test_hashing_embedder_normalised():
    e = HashingEmbedder(dim=128)
    v = e.encode(["a longer sentence about a refund please"])
    assert abs(np.linalg.norm(v[0]) - 1.0) < 1e-5


def test_similar_texts_closer_than_dissimilar():
    e = HashingEmbedder(dim=512)
    v = e.encode([
        "where is my order it hasn't arrived",
        "my order still has not arrived where is it",
        "thank you so much the service was excellent",
    ])
    sim_close = float(v[0] @ v[1])
    sim_far = float(v[0] @ v[2])
    assert sim_close > sim_far


def test_factory_defaults_to_hashing_in_tests():
    e = make_embedder()
    assert "hashing" in e.name
