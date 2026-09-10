"""Retrieve similar *resolved* conversations to ground reply generation.

The corpus is (first customer message) -> (final agent reply) for every
conversation flagged ``resolved``. At query time we embed the incoming message,
take cosine-nearest neighbours, and prefer the **same brand** so the exemplars
carry that brand's voice and policy. If a brand has too few resolved examples we
fall back to cross-brand neighbours and flag it, because a thin same-brand
retrieval is itself an escalation signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import CONFIG
from ..data.normalize import normalize
from ..embeddings import Embedder, make_embedder
from ..types import Conversation, Exemplar


@dataclass
class RetrievalCorpus:
    conv_ids: list[str]
    brands: list[str]
    customer_texts: list[str]
    agent_texts: list[str]
    matrix: np.ndarray  # (n, dim), L2-normalised

    def __len__(self) -> int:
        return len(self.conv_ids)


class ReplyRetriever:
    def __init__(self, corpus: RetrievalCorpus, embedder: Embedder | None = None):
        self.corpus = corpus
        self.embedder = embedder or make_embedder()

    @classmethod
    def build(
        cls, conversations: list[Conversation], embedder: Embedder | None = None
    ) -> ReplyRetriever:
        emb = embedder or make_embedder()
        cids, brands, cust, agent = [], [], [], []
        for c in conversations:
            if not c.resolved:
                continue
            q = c.first_customer_text
            a = c.last_agent_text
            if not q or not a:
                continue
            cids.append(c.conv_id)
            brands.append(c.brand)
            cust.append(normalize(q).clean)
            agent.append(a)
        matrix = (
            emb.encode(cust) if cust else np.zeros((0, getattr(emb, "dim", 384)), np.float32)
        )
        return cls(RetrievalCorpus(cids, brands, cust, agent, matrix), emb)

    def retrieve(
        self, text: str, brand: str, *, k: int | None = None, k_min: int | None = None
    ) -> list[Exemplar]:
        k = k or CONFIG.retrieval.k
        k_min = k_min or CONFIG.retrieval.k_min
        if len(self.corpus) == 0:
            return []
        q = self.embedder.encode([normalize(text).clean])[0]
        sims = self.corpus.matrix @ q  # cosine, both normalised

        brand_mask = np.array([b == brand for b in self.corpus.brands])
        same_brand_hits = int(brand_mask.sum())

        def _topk(candidate_idx: np.ndarray, n: int) -> list[int]:
            if candidate_idx.size == 0:
                return []
            local = np.argsort(-sims[candidate_idx])[:n]
            return candidate_idx[local].tolist()

        chosen: list[int]
        if same_brand_hits >= k_min:
            chosen = _topk(np.where(brand_mask)[0], k)
            if len(chosen) < k:  # top up with cross-brand
                extra = _topk(np.where(~brand_mask)[0], k - len(chosen))
                chosen += extra
        else:
            chosen = _topk(np.argsort(-sims)[: k * 3], k)

        out = []
        for idx in chosen:
            out.append(
                Exemplar(
                    conv_id=self.corpus.conv_ids[idx],
                    brand=self.corpus.brands[idx],
                    customer_text=self.corpus.customer_texts[idx],
                    agent_text=self.corpus.agent_texts[idx],
                    similarity=float(sims[idx]),
                    same_brand=self.corpus.brands[idx] == brand,
                )
            )
        out.sort(key=lambda e: -e.similarity)
        return out
