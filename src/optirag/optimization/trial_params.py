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


def index_cache_key(
    *,
    corpus_version: str,
    chunk_strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    cleaning_mode: str,
    embedding_model: str,
    output_dim: int,
    pinecone_metric: str,
    l2_normalize: bool,
) -> str:
    """Stable id for a Pinecone index (same corpus embedded the same way)."""
    h = hashlib.sha256(
        json.dumps(
            {
                "corpus_version": corpus_version,
                "chunk_strategy": chunk_strategy,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
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
