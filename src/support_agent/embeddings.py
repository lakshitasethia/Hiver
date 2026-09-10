"""Text -> vector, with a local model by default and a zero-download fallback.

``SentenceTransformerEmbedder`` uses ``all-MiniLM-L6-v2`` (384-d), runs on CPU,
needs no API key. ``HashingEmbedder`` is a deterministic hashed char-n-gram
vector used when ``SUPPORT_AGENT_EMBEDDER=hashing`` — lower quality but no
download and byte-identical across machines, which is what CI and reproducible
tests need.

Both cache to ``.cache/emb`` keyed by ``sha1(backend + model + text)`` so
re-runs of taxonomy/train/eval are fast.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from .config import CACHE_DIR, CONFIG


@runtime_checkable
class Embedder(Protocol):
    dim: int
    name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        ...


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class HashingEmbedder:
    """Hashed bag of character 3/4/5-grams, L2-normalised. Deterministic."""

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim or CONFIG.embedding.hashing_dim
        self.name = f"hashing-{self.dim}"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        t = re.sub(r"\s+", " ", (text or "").lower()).strip()
        t = f" {t} "
        for n in (3, 4, 5):
            for i in range(len(t) - n + 1):
                gram = t[i : i + n]
                h = int.from_bytes(hashlib.md5(gram.encode()).digest()[:8], "little")
                sign = 1.0 if (h >> 63) & 1 else -1.0
                v[h % self.dim] += sign
        return v

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        mat = np.vstack([self._vec(t) for t in texts]) if len(texts) else np.zeros((0, self.dim), np.float32)
        return _l2_normalize(mat)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or CONFIG.embedding.st_model
        self._model = SentenceTransformer(self.model_name)
        # method was renamed across sentence-transformers versions
        get_dim = getattr(self._model, "get_embedding_dimension", None) or (
            self._model.get_sentence_embedding_dimension
        )
        self.dim = get_dim()
        self.name = self.model_name.split("/")[-1]

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not len(texts):
            return np.zeros((0, self.dim), np.float32)
        mat = self._model.encode(
            list(texts), convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        return mat.astype(np.float32)


class CachingEmbedder:
    """Wraps any embedder with an on-disk per-text cache."""

    def __init__(self, inner: Embedder, cache_dir: Path | None = None) -> None:
        self.inner = inner
        self.dim = inner.dim
        self.name = inner.name
        self.dir = (cache_dir or (CACHE_DIR / "emb")) / self.name
        self.dir.mkdir(parents=True, exist_ok=True)

    def _key(self, text: str) -> str:
        return hashlib.sha1(f"{self.name}\x00{text}".encode()).hexdigest()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        out: list[np.ndarray | None] = [None] * len(texts)
        missing_idx: list[int] = []
        for i, t in enumerate(texts):
            fp = self.dir / f"{self._key(t)}.npy"
            if fp.exists():
                out[i] = np.load(fp)
            else:
                missing_idx.append(i)
        if missing_idx:
            fresh = self.inner.encode([texts[i] for i in missing_idx])
            for j, i in enumerate(missing_idx):
                vec = fresh[j]
                np.save(self.dir / f"{self._key(texts[i])}.npy", vec)
                out[i] = vec
        return np.vstack(out) if out else np.zeros((0, self.dim), np.float32)


def make_embedder(backend: str | None = None, *, cache: bool | None = None) -> Embedder:
    backend = backend or CONFIG.embedding.backend
    cache = CONFIG.embedding.cache if cache is None else cache
    if backend == "hashing":
        emb: Embedder = HashingEmbedder()
    elif backend in ("st", "sentence-transformers"):
        emb = SentenceTransformerEmbedder()
    else:
        raise ValueError(f"unknown embedder backend: {backend!r}")
    return CachingEmbedder(emb) if cache else emb
