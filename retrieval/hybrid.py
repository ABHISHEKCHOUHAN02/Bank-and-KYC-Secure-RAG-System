"""
Hybrid retriever: dense (Chroma + bge) + sparse (BM25) fused with weighted RRF.

Usage:
    from retrieval.hybrid import get_retriever
    retriever = get_retriever()                       # loaded once, reused
    results = retriever.retrieve("What is the KYC periodic updation rule?")
    baseline = retriever.retrieve(q, mode="dense")    # plain-RAG baseline for the eval

Quick manual test (from repo root):
    python -m retrieval.hybrid "your question here"

Requires the indexes built by `python -m retrieval.ingest`.
"""

from __future__ import annotations

import logging
import pickle
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

import numpy as np

from app.config import settings
from app.schemas import Chunk, RetrievedChunk
from retrieval.ingest import load_chunks, tokenize
from retrieval.Rrf import reciprocal_rank_fusion  # file must be named rrf.py (lowercase)

logger = logging.getLogger(__name__)

Mode = Literal["dense", "bm25", "hybrid"]

# --- Tunables (move to app/config.py once you settle on values) -------------
RRF_K = 20                      # lower than 60 so rank position matters more
BM25_FETCH_K = 10               # shorter sparse list: weak BM25 matches stop earning points
WEIGHTS_CODE = (0.6, 1.0)       # (dense, bm25) when the query has code-like tokens
WEIGHTS_NATURAL = (1.0, 0.3)    # (dense, bm25) for natural-language queries

# Query-side only. The index is untouched; dropped tokens simply stop scoring.
STOPWORDS = frozenset(
    """a an the is are was were be been am do does did i me my we our you your it its
    to of in on at for by with from if will would can could should may might any some
    and or but what which who whom how when where why this that these those there then
    than so as about into over after before have has had within tell get give""".split()
)


def is_code_like(query: str) -> bool:
    """True if the query contains an ID/code/email/domain-style token (E47, AFB-DISP-01,
    RBI/2022-23/111, pno@bank.com, cms.rbi.org.in). Plain numbers like '25,000' don't count."""
    for w in query.split():
        if "@" in w:
            return True
        if re.search(r"[A-Za-z]", w) and re.search(r"\d", w):
            return True
        if re.search(r"\w+\.[A-Za-z]{2,}\b", w):
            return True
    return False


def _bm25_tokens(query: str) -> list[str]:
    toks = tokenize(query)
    kept = [t for t in toks if any(c.isdigit() for c in t) or t not in STOPWORDS]
    return kept or toks  # never send an empty query


@dataclass
class RetrievalDebug:
    """Per-retriever candidate lists, useful for the trace and for debugging."""

    dense_ids: list[str] = field(default_factory=list)
    bm25_ids: list[str] = field(default_factory=list)
    code_like: bool = False
    weights: tuple[float, float] = (1.0, 1.0)


class HybridRetriever:
    def __init__(self) -> None:
        import chromadb
        from sentence_transformers import SentenceTransformer

        # Chunks (source of truth for text + metadata)
        if not settings.chunks_path.exists():
            raise RuntimeError(
                f"{settings.chunks_path} not found. Run `python -m retrieval.ingest` first."
            )
        chunks = load_chunks()
        self._chunks: dict[str, Chunk] = {c.chunk_id: c for c in chunks}

        # BM25 (pickle: only load files you built yourself)
        if not settings.bm25_path.exists():
            raise RuntimeError(f"{settings.bm25_path} not found. Run `python -m retrieval.ingest`.")
        with settings.bm25_path.open("rb") as f:
            payload = pickle.load(f)
        self._bm25 = payload["bm25"]
        self._bm25_ids: list[str] = payload["chunk_ids"]

        # Chroma
        client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        try:
            self._collection = client.get_collection(settings.chroma_collection)
        except Exception as exc:
            raise RuntimeError(
                f"Chroma collection '{settings.chroma_collection}' not found. "
                "Run `python -m retrieval.ingest`."
            ) from exc

        # Consistency check: all three stores must come from the same ingest run.
        n_chroma = self._collection.count()
        if not (len(self._chunks) == len(self._bm25_ids) == n_chroma):
            raise RuntimeError(
                f"Index mismatch: chunks.jsonl={len(self._chunks)}, "
                f"bm25={len(self._bm25_ids)}, chroma={n_chroma}. Re-run ingest."
            )

        self._embedder = SentenceTransformer(settings.embedding_model)
        logger.info("HybridRetriever ready: %d chunks", len(self._chunks))

    # ------------------------------------------------------------------
    # Individual retrievers: each returns [(chunk_id, raw_score)], best first
    # ------------------------------------------------------------------
    def dense_search(self, query: str, k: int) -> list[tuple[str, float]]:
        # Query gets the BGE instruction prefix; passages were embedded without it.
        emb = self._embedder.encode(
            settings.query_instruction + query,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        res = self._collection.query(
            query_embeddings=[emb],
            n_results=min(k, len(self._chunks)),
            include=["distances"],
        )
        ids = res["ids"][0]
        distances = res["distances"][0]
        # Collection uses cosine space, so distance = 1 - cosine similarity.
        return [(cid, 1.0 - float(d)) for cid, d in zip(ids, distances)]

    def embed_query(self, query: str) -> np.ndarray:
        """Return the normalized BGE query embedding used by dense retrieval."""
        vector = self._embedder.encode(
            settings.query_instruction + query,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vector, dtype=np.float32)

    def bm25_search(self, query: str, k: int) -> list[tuple[str, float]]:
        tokens = _bm25_tokens(query)
        if not tokens:
            return []
        scores = np.asarray(self._bm25.get_scores(tokens))
        top = np.argsort(-scores)[:k]
        return [(self._bm25_ids[i], float(scores[i])) for i in top if scores[i] > 0]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def retrieve_debug(
        self,
        query: str,
        top_k: int | None = None,
        fetch_k: int | None = None,
        mode: Mode = "hybrid",
    ) -> tuple[list[RetrievedChunk], RetrievalDebug]:
        top_k = top_k or settings.top_k
        fetch_k = fetch_k or settings.fetch_k
        query = query.strip()
        debug = RetrievalDebug()
        if not query:
            return [], debug

        # Hybrid uses a short BM25 list; the standalone BM25 baseline uses the full fetch_k.
        sparse_k = BM25_FETCH_K if mode == "hybrid" else fetch_k

        dense: list[tuple[str, float]] = []
        sparse: list[tuple[str, float]] = []
        if mode in ("dense", "hybrid"):
            dense = self.dense_search(query, fetch_k)
            debug.dense_ids = [cid for cid, _ in dense]
        if mode in ("bm25", "hybrid"):
            sparse = self.bm25_search(query, sparse_k)
            debug.bm25_ids = [cid for cid, _ in sparse]

        if mode == "dense":
            ranked, label = dense[:top_k], "dense"
        elif mode == "bm25":
            ranked, label = sparse[:top_k], "bm25"
        else:
            debug.code_like = is_code_like(query)
            debug.weights = WEIGHTS_CODE if debug.code_like else WEIGHTS_NATURAL
            ranked = reciprocal_rank_fusion(
                [debug.dense_ids, debug.bm25_ids],
                k=RRF_K,
                weights=debug.weights,
                top_n=top_k,
            )
            label = "rrf"

        results = [
            RetrievedChunk(chunk=self._chunks[cid], score=float(score), rank=i, retriever=label)
            for i, (cid, score) in enumerate(ranked, start=1)
        ]
        return results, debug

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        fetch_k: int | None = None,
        mode: Mode = "hybrid",
    ) -> list[RetrievedChunk]:
        return self.retrieve_debug(query, top_k, fetch_k, mode)[0]


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    """Load indexes and the embedding model once per process."""
    return HybridRetriever()


if __name__ == "__main__":
    import sys

    # Silence library chatter BEFORE the model loads.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    for name in ("httpx", "httpcore", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(name).setLevel(logging.ERROR)

    q = " ".join(sys.argv[1:]) or "What documents are required for KYC?"
    r = get_retriever()
    print(f"Loaded {len(r._chunks)} chunks")

    for mode in ("dense", "bm25", "hybrid"):
        hits, dbg = r.retrieve_debug(q, mode=mode)
        d_rank = {cid: i for i, cid in enumerate(dbg.dense_ids, 1)}
        b_rank = {cid: i for i, cid in enumerate(dbg.bm25_ids, 1)}
        extra = f" (code_like={dbg.code_like}, weights dense/bm25={dbg.weights})" if mode == "hybrid" else ""
        print(f"\n=== {mode}{extra} ===")
        for hit in hits:
            m = hit.chunk.metadata
            page = f" | p.{m.page}" if m.page is not None else ""
            ranks = ""
            if mode == "hybrid":
                cid = hit.chunk.chunk_id
                ranks = f"  [dense#{d_rank.get(cid, '-')} bm25#{b_rank.get(cid, '-')}]"
            print(
                f"{hit.rank}. [{hit.score:.4f}] {m.doc} | {m.circular_no or '-'} | "
                f"{m.section or '-'}{page}{ranks}\n   {hit.chunk.text[:120]!r}"
            )
