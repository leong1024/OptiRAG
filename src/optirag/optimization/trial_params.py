from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from typing import Any, Literal

from optirag.config.embedding_registry import get_embedding_dim

ChunkStrategy = Literal[
    "identity_one_vec_per_line",
    "fixed_window",
    "sliding_window",
    "recursive",
]
CleaningMode = Literal["none", "light_normalize"]
PineconeMetric = Literal["cosine", "dotproduct", "euclidean"]
ParentDedup = Literal["off", "keep_highest_score"]


@dataclass
class Stage1TrialParams:
    """Bounded Stage-1 search space (plan §4.1)."""

    # Preprocess
    cleaning_mode: CleaningMode = "none"
    # Chunking
    chunk_strategy: ChunkStrategy = "identity_one_vec_per_line"
    chunk_size: int = 1024
    chunk_overlap: int = 0
    min_chunk_chars: int = 0
    # Embedding
    embedding_model: str = "gemini-embedding-001"
    output_dim_override: int | None = None
    l2_normalize: bool = True
    # Pinecone index + query
    pinecone_metric: PineconeMetric = "cosine"
    top_k: int = 10
    # Post-retrieval (app)
    min_similarity: float | None = None
    max_chunks_per_beir_id: int = 2
    context_char_budget: int = 8000
    parent_dedup_policy: ParentDedup = "keep_highest_score"
    # Optional
    rerank_enabled: bool = False
    rerank_m: int = 20

    def embedding_dim(self) -> int:
        return get_embedding_dim(self.embedding_model, self.output_dim_override)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Stage1TrialParams:
        allowed = {f.name for f in fields(cls)}
        clean = {k: v for k, v in d.items() if k in allowed}
        return cls(**clean)


def trial_params_fingerprint(p: Stage1TrialParams) -> str:
    raw = json.dumps(p.to_json_dict(), sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


INDEX_CACHE_KEY_VERSION = 1


def index_cache_key(
    *,
    corpus_version: str,
    chunk_strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_chars: int,
    cleaning_mode: str,
    embedding_model: str,
    output_dim: int,
    pinecone_metric: str,
    l2_normalize: bool,
) -> str:
    """Stable id for a logical index (chunk + embed + metric); use as Pinecone namespace base."""
    h = hashlib.sha256(
        json.dumps(
            {
                "v": INDEX_CACHE_KEY_VERSION,
                "corpus_version": corpus_version,
                "chunk_strategy": chunk_strategy,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "min_chunk_chars": min_chunk_chars,
                "cleaning_mode": cleaning_mode,
                "embedding_model": embedding_model,
                "output_dim": output_dim,
                "pinecone_metric": pinecone_metric,
                "l2_normalize": l2_normalize,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:20]
    safe_model = embedding_model.replace(".", "-").replace("_", "-")
    return f"optirag-{safe_model}-d{output_dim}-m{pinecone_metric[:3]}-n{int(l2_normalize)}-{h}"


def index_cache_key_from_params(corpus_version: str, p: Stage1TrialParams) -> str:
    """Convenience: full key from trial params (index-defining fields only)."""
    return index_cache_key(
        corpus_version=corpus_version,
        chunk_strategy=p.chunk_strategy,
        chunk_size=p.chunk_size,
        chunk_overlap=p.chunk_overlap,
        min_chunk_chars=p.min_chunk_chars,
        cleaning_mode=p.cleaning_mode,
        embedding_model=p.embedding_model,
        output_dim=p.embedding_dim(),
        pinecone_metric=p.pinecone_metric,
        l2_normalize=p.l2_normalize,
    )


def pinecone_namespace_id(fingerprint: str) -> str:
    """Pinecone namespace: short stable id (under typical length limits)."""
    return f"opt-{fingerprint}"


def stage1_index_fingerprint(corpus_version: str, p: Stage1TrialParams) -> str:
    """Stable 16-char id for index-defining params (excludes top_k, post-retrieval)."""
    key = index_cache_key_from_params(corpus_version, p)
    return hashlib.sha256(key.encode()).hexdigest()[:16]
