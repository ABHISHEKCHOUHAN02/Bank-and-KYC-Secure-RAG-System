"""
Central configuration for bank-rag.

Every tunable lives here. Other modules should import from this file
(`from app.config import settings`) and never hardcode paths, model names,
thresholds, or allowlists.

Secrets (e.g. GEMINI_API_KEY) are read from environment variables / .env,
never hardcoded. Non-secret tunables can be overridden via env vars too,
which makes ablation runs easy (e.g. TOP_K=3 python eval/run_eval.py).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()  # read .env if present


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# Repo root = parent of the `app/` directory this file lives in.
BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:

    # Paths
    
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    raw_dir: Path = BASE_DIR / "data" / "raw"
    raw_rbi_dir: Path = BASE_DIR / "data" / "raw" / "rbi"
    raw_fictional_dir: Path = BASE_DIR / "data" / "raw" / "fictional_bank"
    index_dir: Path = BASE_DIR / "data" / "indexes"
    chroma_dir: Path = BASE_DIR / "data" / "indexes" / "chroma"
    bm25_path: Path = BASE_DIR / "data" / "indexes" / "bm25.pkl"
    chunks_path: Path = BASE_DIR / "data" / "indexes" / "chunks.jsonl"
    manifest_path: Path = BASE_DIR / "data" / "manifest.json"
    eval_qa_path: Path = BASE_DIR / "eval" / "qa_set.jsonl"
    eval_results_dir: Path = BASE_DIR / "eval" / "results"


    # Embeddings

    
    
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
   
    # bge-small-en-v1.5 truncates at 512 tokens; verify on its model card
    # if you swap models.
    
    
    embedding_max_tokens: int = _env_int("EMBEDDING_MAX_TOKENS", 512)
    

    query_instruction: str = os.getenv(
        "QUERY_INSTRUCTION",
        "Represent this sentence for searching relevant passages: ",
    )

  
    # Chunking (measured in TOKENS of the embedding model's tokenizer)
    # ~400 tokens leaves headroom under the 512 limit for special tokens
    # and for metadata/prefix text if you prepend any to the chunk.
    
    
    chunk_size_tokens: int = _env_int("CHUNK_SIZE_TOKENS", 400)
    chunk_overlap_tokens: int = _env_int("CHUNK_OVERLAP_TOKENS", 60)  # 15%


    # Retrieval
 
    fetch_k: int = _env_int("FETCH_K", 20)  # candidates pulled from EACH retriever
    top_k: int = _env_int("TOP_K", 5)  # chunks passed to the generator after fusion
    rrf_k: int = _env_int("RRF_K", 60)  # RRF constant: score = 1 / (rrf_k + rank)
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "bank_kyc_rag")


    # LLM (Gemini). Model names change often: verify against Google's docs.

    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    llm_temperature: float = _env_float("LLM_TEMPERATURE", 0.0)
    llm_max_retries: int = _env_int("LLM_MAX_RETRIES", 3)  # API-level retries
    # Cheaper model for grader/rewriter. Defaults to the main model; override once --list shows a lighter one.
    gemini_fast_model: str = os.getenv("GEMINI_FAST_MODEL", os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"))
    llm_requests_per_minute: int = _env_int("LLM_RPM", 10)  # client-side pacing
    llm_cache_enabled: bool = _env_bool("LLM_CACHE", False)  # disk cache: evals only
    llm_cache_dir: Path = BASE_DIR / "data" / "llm_cache"

  
    # LangGraph self-healing loop - it can retry the LLM with a new prompt if the output fails validation.
  
    max_rewrite_retries: int = _env_int("MAX_REWRITE_RETRIES", 2)
    enable_web_fallback: bool = _env_bool("ENABLE_WEB_FALLBACK", True)

   
    # Web fallback allowlist (enforce by filtering result URLs, not just
    # by adding site: operators to the query)

    web_allowlist: tuple[str, ...] = ("rbi.org.in", "sebi.gov.in", "npci.org.in")
    web_max_results: int = _env_int("WEB_MAX_RESULTS", 5)

    # semantic caching of embeddings and retrieval results. This is a big speedup for repeated queries.
    
    
    cache_enabled: bool = _env_bool("CACHE_ENABLED", True)
    cache_similarity_threshold: float = _env_float("CACHE_SIMILARITY_THRESHOLD", 0.92)
    cache_ttl_seconds: int = _env_int("CACHE_TTL_SECONDS", 24 * 60 * 60)
    redis_url: str = os.getenv("REDIS_URL", "")  # empty => in-memory cache


    # Security toggles (so ablations can switch layers on/off)
  
    enable_injection_filter: bool = _env_bool("ENABLE_INJECTION_FILTER", True)
    enable_pii_redaction: bool = _env_bool("ENABLE_PII_REDACTION", True)
    enable_scope_guard: bool = _env_bool("ENABLE_SCOPE_GUARD", True)
    enable_output_guard: bool = _env_bool("ENABLE_OUTPUT_GUARD", True)

    def validate(self) -> None:
        """Fail fast on inconsistent settings."""
        # +2 reserves room for [CLS]/[SEP]-style special tokens.
        
        if self.chunk_size_tokens + 2 > self.embedding_max_tokens:
            raise ValueError(
                f"chunk_size_tokens ({self.chunk_size_tokens}) is too close to or above "
                f"the embedding model limit ({self.embedding_max_tokens}); "
                "chunks would be silently truncated."
            )
        if not 0 <= self.chunk_overlap_tokens < self.chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be >= 0 and < chunk_size_tokens")
        if self.top_k > self.fetch_k:
            raise ValueError("top_k must be <= fetch_k")
        if self.max_rewrite_retries < 0:
            raise ValueError("max_rewrite_retries must be >= 0")
        if self.web_max_results < 1:
            raise ValueError("web_max_results must be >= 1")
        if not 0.0 < self.cache_similarity_threshold <= 1.0:
            raise ValueError("cache_similarity_threshold must be in (0, 1)")
        if self.cache_ttl_seconds < 1:
            raise ValueError("cache_ttl_seconds must be >= 1")

    def ensure_dirs(self) -> None:
        """Create writable directories. Call from scripts/entrypoints, not on import."""
        for d in (
            self.raw_rbi_dir,
            self.raw_fictional_dir,
            self.index_dir,
            self.eval_results_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.validate()
