"""Frozen system prompts for Stage 1 Optuna (query + answer paths)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FrozenStage1Prompts:
    """Identity query path: rewriter returns the user query unchanged."""

    query_system: str = (
        "You pass through the user query unchanged for retrieval. Output only the query text, no preamble."
    )
    answer_system: str = (
        "You are a careful assistant. Answer using only the provided context. "
        "If the context is insufficient, say you do not know. Cite passages briefly when possible."
    )


def identity_retrieval_query(user_query: str) -> str:
    """No LLM call: use raw query for embedding (preferred ablation for Stage 1)."""
    return user_query.strip()
