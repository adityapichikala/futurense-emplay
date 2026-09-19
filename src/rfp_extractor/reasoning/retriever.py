"""Hybrid retrieval: BM25 (exact identifiers) + TF-IDF vectors (semantics).

Why hybrid, and why it matters for RFPs:

* **BM25** nails exact identifiers - bid numbers, SKUs, "Closing Date".  A dense
  embedding frequently misses ``JA-207652`` because it has no semantic content;
  BM25 treats it as a rare token and ranks it first.
* **TF-IDF cosine** handles paraphrase - the query "when is the proposal due"
  against text that says "Solicitation Due".

Scores are min-max normalized per index and combined with configurable weights,
which is the standard RRF-lite fusion used in production RAG.

The vector backend defaults to a zero-dependency TF-IDF implementation so the
pipeline runs offline and deterministically.  If ``sentence-transformers`` is
installed and configured, ``SentenceTransformerBackend`` can be swapped in
without touching the retriever.
"""

from __future__ import annotations

import logging
import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from .chunking import Chunk

log = logging.getLogger(__name__)

#: Tail classes include '.' so hostnames and file names stay whole, but a
#: trailing period must be stripped or "JA-207652." never matches "JA-207652".
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-_.#/]*")

_TRAILING_PUNCT = ".,;:!?"

_STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "was", "were", "will", "with", "your", "you", "our", "their", "not", "any", "all", "may", "must", "shall", "should"]
)


def tokenize(text: str) -> list[str]:
    raw = (t.strip(_TRAILING_PUNCT).lower() for t in _TOKEN_RE.findall(text))
    return [t for t in raw if t not in _STOPWORDS and len(t) > 1]


# ---------------------------------------------------------------------------
# Vector backends
# ---------------------------------------------------------------------------


class VectorBackend(ABC):
    @abstractmethod
    def fit(self, texts: list[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def encode_query(self, text: str) -> np.ndarray:
        raise NotImplementedError

    @property
    @abstractmethod
    def matrix(self) -> np.ndarray:
        raise NotImplementedError


class TfidfBackend(VectorBackend):
    """Deterministic, dependency-free TF-IDF vectors."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self._matrix: np.ndarray | None = None

    def fit(self, texts: list[str]) -> None:
        docs_tokens = [tokenize(t) for t in texts]
        vocab: dict[str, int] = {}
        for tokens in docs_tokens:
            for tok in tokens:
                if tok not in vocab:
                    vocab[tok] = len(vocab)
        self.vocab = vocab
        n = len(docs_tokens)
        df: Counter[str] = Counter()
        for tokens in docs_tokens:
            for tok in set(tokens):
                df[tok] += 1
        self.idf = np.array(
            [math.log((1 + n) / (1 + df.get(tok, 0))) + 1.0 for tok in vocab], dtype=np.float64
        )
        matrix = np.zeros((n, len(vocab)), dtype=np.float64)
        for i, tokens in enumerate(docs_tokens):
            counts = Counter(tokens)
            for tok, count in counts.items():
                j = vocab[tok]
                matrix[i, j] = (1.0 + math.log(count)) * self.idf[j]
        # L2-normalize rows
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix = matrix / norms

    def encode_query(self, text: str) -> np.ndarray:
        assert self.idf is not None
        vec = np.zeros(len(self.vocab), dtype=np.float64)
        counts = Counter(tokenize(text))
        for tok, count in counts.items():
            j = self.vocab.get(tok)
            if j is None:
                continue
            vec[j] = (1.0 + math.log(count)) * self.idf[j]
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    @property
    def matrix(self) -> np.ndarray:
        assert self._matrix is not None, "backend not fitted"
        return self._matrix


class SentenceTransformerBackend(VectorBackend):
    """Optional dense-embedding backend (requires ``sentence-transformers``)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self.model = SentenceTransformer(model_name)
        self._matrix: np.ndarray | None = None

    def fit(self, texts: list[str]) -> None:
        self._matrix = np.asarray(self.model.encode(texts, normalize_embeddings=True))

    def encode_query(self, text: str) -> np.ndarray:
        return np.asarray(self.model.encode([text], normalize_embeddings=True))[0]

    @property
    def matrix(self) -> np.ndarray:
        assert self._matrix is not None, "backend not fitted"
        return self._matrix


# ---------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------


@dataclass
class RetrievalHit:
    chunk: Chunk
    score: float
    bm25_score: float
    vector_score: float


class HybridRetriever:
    """Fuse BM25 and vector scores over a chunk index."""

    def __init__(
        self,
        chunks: list[Chunk],
        *,
        backend: VectorBackend | None = None,
        bm25_weight: float = 0.6,
        vector_weight: float = 0.4,
    ) -> None:
        self.chunks = chunks
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks]) if chunks else None
        self.backend = backend or TfidfBackend()
        if chunks:
            self.backend.fit([c.text for c in chunks])

    def search(
        self,
        query: str,
        *,
        top_k: int = 6,
        doc_types: set[str] | None = None,
        boost_precedence: bool = True,
    ) -> list[RetrievalHit]:
        if not self.chunks:
            return []

        # rank_bm25 omits the "+1" IDF smoothing used by Lucene, so terms present
        # in most chunks of a small index score *negative*.  Clamping is standard
        # practice and prevents a matching chunk from being ranked below zero.
        if self.bm25 is None:
            return []
        bm25_scores = np.maximum(
            np.asarray(self.bm25.get_scores(tokenize(query)), dtype=np.float64), 0.0
        )
        query_vec = self.backend.encode_query(query)
        matrix = self.backend.matrix
        vector_scores = matrix @ query_vec if matrix.size else np.zeros(len(self.chunks))

        def norm(arr: np.ndarray) -> np.ndarray:
            lo, hi = float(arr.min()), float(arr.max())
            return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)

        combined = self.bm25_weight * norm(bm25_scores) + self.vector_weight * norm(vector_scores)
        if float(combined.max()) <= 0:
            # Degenerate case A: every chunk scored identically, so min-max
            # normalization flattened everything to zero - use raw scores.
            combined = bm25_scores + np.maximum(vector_scores, 0.0)
        if float(combined.max()) <= 0:
            # Degenerate case B: no lexical or vector overlap at all (tiny
            # synthetic indexes).  Give every chunk an equal base score so the
            # precedence boost below still orders addenda above the master RFP.
            combined = np.ones(len(self.chunks), dtype=np.float64)

        hits: list[RetrievalHit] = []
        for i, chunk in enumerate(self.chunks):
            if doc_types and chunk.doc_type not in doc_types:
                continue
            score = float(combined[i])
            if boost_precedence:
                # addenda/portal notices are more likely to hold the current truth
                score *= 1.0 + min(chunk.precedence, 150) / 500.0
            if score <= 0:
                continue
            hits.append(
                RetrievalHit(
                    chunk=chunk,
                    score=score,
                    bm25_score=float(bm25_scores[i]),
                    vector_score=float(vector_scores[i]),
                )
            )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def blocks_for(self, hits: list[RetrievalHit]) -> set[str]:
        block_ids: set[str] = set()
        for hit in hits:
            block_ids.update(hit.chunk.block_ids)
        return block_ids
