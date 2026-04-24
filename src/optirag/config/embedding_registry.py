"""Default output dimensions for Gemini embedding models (override via experiment YAML)."""

from __future__ import annotations

# Values align with common Google embedding API output_dimensionality; confirm in your project.
EMBEDDING_MODEL_DIM_DEFAULT: dict[str, int] = {
    "gemini-embedding-001": 3072,
    "gemini-embedding-002": 3072,
}


def get_embedding_dim(model: str, override: int | None) -> int:
    if override is not None:
        return override
    return EMBEDDING_MODEL_DIM_DEFAULT.get(model, 3072)
