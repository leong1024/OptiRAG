from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class DataSplit(StrEnum):
    """BEIR-style split for FiQA."""

    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class Passage:
    """A BEIR corpus line (one doc id = one line)."""

    beir_corpus_id: str
    text: str


@dataclass(frozen=True, slots=True)
class TextChunk:
    """Chunk derived from a passage; all chunks carry the parent id."""

    beir_corpus_id: str
    text: str
    chunk_index: int
    extra_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class QueryRecord:
    query_id: str
    text: str


@dataclass(frozen=True, slots=True)
class RetrievedPassage:
    beir_corpus_id: str
    text: str
    score: float
    vector_id: str | None = None


@dataclass
class RAGResult:
    query_id: str
    user_query: str
    retrieval_query: str
    retrieved: list[RetrievedPassage]
    answer: str
    contexts_for_eval: list[str]
    parent_ids_in_context: list[str]
