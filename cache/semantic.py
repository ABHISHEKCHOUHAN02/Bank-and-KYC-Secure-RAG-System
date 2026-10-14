"""TTL-based semantic response cache with memory and optional Redis backends.

Only privacy-processed queries and guarded, non-refusal responses should be passed
to this module. Query text is not persisted: Redis and memory store embeddings and
safe response payloads only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from uuid import uuid4

import numpy as np

from app.config import settings
from app.schemas import QueryResponse

logger = logging.getLogger(__name__)


@dataclass
class CacheHit:
    payload: dict
    similarity: float


@dataclass
class _MemoryEntry:
    embedding: list[float]
    payload: dict
    expires_at: float


class SemanticCache:
    """Find the nearest unexpired query embedding above a cosine threshold."""

    def __init__(self, *, ttl_seconds: int | None = None,
                 similarity_threshold: float | None = None, max_entries: int = 500) -> None:
        self.ttl_seconds = ttl_seconds or settings.cache_ttl_seconds
        self.similarity_threshold = similarity_threshold or settings.cache_similarity_threshold
        self.max_entries = max_entries
        self._lock = threading.RLock()
        self._memory: list[_MemoryEntry] = []
        self._redis = None
        self._redis_unavailable = False
        self._namespace = self._make_namespace()

    @staticmethod
    def _make_namespace() -> str:
        try:
            stat = settings.chunks_path.stat()
            index_version = f"{stat.st_size}:{stat.st_mtime_ns}"
        except OSError:
            index_version = "no-index"
        value = "|".join((settings.embedding_model, settings.query_instruction,
                          settings.chroma_collection, index_version))
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]

    @property
    def _redis_prefix(self) -> str:
        return f"bankguard:semantic:{self._namespace}:"

    def _get_redis(self):
        if not settings.redis_url or self._redis_unavailable:
            return None
        if self._redis is None:
            try:
                import redis

                self._redis = redis.Redis.from_url(
                    settings.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=1.5,
                    socket_timeout=1.5,
                )
                self._redis.ping()
            except Exception as exc:
                self._redis_unavailable = True
                self._redis = None
                logger.warning("Redis semantic cache unavailable (%s); using in-memory cache.",
                               type(exc).__name__)
        return self._redis

    @staticmethod
    def _embed(query: str) -> list[float]:
        # Reuse the retriever's loaded BGE model rather than loading a second copy.
        from retrieval.hybrid import get_retriever

        retriever = get_retriever()
        return retriever.embed_query(query).tolist()

    def _iter_redis_entries(self, client):
        for key in client.scan_iter(match=f"{self._redis_prefix}*", count=200):
            try:
                raw = client.get(key)
                if raw:
                    yield json.loads(raw)
            except Exception as exc:
                logger.warning("Skipping invalid Redis cache entry (%s).", type(exc).__name__)

    def lookup(self, query: str) -> CacheHit | None:
        embedding = np.asarray(self._embed(query), dtype=np.float32)
        best: CacheHit | None = None
        now = time.time()

        with self._lock:
            self._memory = [entry for entry in self._memory if entry.expires_at > now]
            entries: list[tuple[list[float], dict]] = [
                (entry.embedding, entry.payload) for entry in self._memory
            ]

        client = self._get_redis()
        if client is not None:
            try:
                for row in self._iter_redis_entries(client):
                    entries.append((row["embedding"], row["payload"]))
            except Exception as exc:
                self._redis_unavailable = True
                self._redis = None
                logger.warning("Redis semantic lookup failed (%s); using in-memory entries.",
                               type(exc).__name__)

        for candidate, payload in entries:
            candidate_vector = np.asarray(candidate, dtype=np.float32)
            if candidate_vector.shape != embedding.shape:
                continue
            similarity = float(np.dot(embedding, candidate_vector))
            if similarity >= self.similarity_threshold and (best is None or similarity > best.similarity):
                best = CacheHit(payload=payload, similarity=similarity)
        return best

    def store(self, query: str, response: QueryResponse) -> None:
        if response.refused:
            return
        embedding = self._embed(query)
        # Trace includes request-specific PII entity types and model execution details.
        # It is intentionally excluded from cache storage.
        payload = response.model_dump(mode="json", exclude={"trace"})
        record = {"embedding": embedding, "payload": payload}
        expiry = time.time() + self.ttl_seconds

        with self._lock:
            self._memory.append(_MemoryEntry(embedding, payload, expiry))
            if len(self._memory) > self.max_entries:
                self._memory = self._memory[-self.max_entries:]

        client = self._get_redis()
        if client is not None:
            try:
                client.setex(
                    f"{self._redis_prefix}{uuid4().hex}",
                    self.ttl_seconds,
                    json.dumps(record, ensure_ascii=False),
                )
            except Exception as exc:
                self._redis_unavailable = True
                self._redis = None
                logger.warning("Redis semantic write failed (%s); kept in memory.",
                               type(exc).__name__)


@lru_cache(maxsize=1)
def get_semantic_cache() -> SemanticCache:
    return SemanticCache()
