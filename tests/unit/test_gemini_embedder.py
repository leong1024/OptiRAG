from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from optirag.adapters.gemini.embedder import GeminiEmbedder


class _FakeModels:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def embed_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        contents = kwargs["contents"]
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[float(i)]) for i, _ in enumerate(contents)])


def _embedder(models: _FakeModels) -> GeminiEmbedder:
    embedder = object.__new__(GeminiEmbedder)
    embedder.model_id = "gemini-embedding-001"
    embedder.output_dimensionality = None
    embedder.l2_normalize = False
    embedder.max_retries = 1
    embedder.max_rate_limit_retries = 0
    embedder._client = SimpleNamespace(models=models)
    return embedder


def test_embed_documents_filters_empty_contents() -> None:
    models = _FakeModels()
    embedder = _embedder(models)

    vectors = embedder.embed_documents(["", "alpha", "   ", "beta"])

    assert vectors == [[0.0], [1.0]]
    assert models.calls[0]["contents"] == ["alpha", "beta"]


def test_embed_documents_skips_provider_when_all_contents_empty() -> None:
    models = _FakeModels()
    embedder = _embedder(models)

    assert embedder.embed_documents(["", "   "]) == []
    assert models.calls == []


def test_embed_query_rejects_empty_text() -> None:
    embedder = _embedder(_FakeModels())

    with pytest.raises(ValueError, match="empty query"):
        embedder.embed_query("   ")
