from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

if "pinecone" not in sys.modules:
    pinecone_stub = types.ModuleType("pinecone")
    pinecone_stub.Pinecone = object
    sys.modules["pinecone"] = pinecone_stub

from optirag.data.beir_fiqa import FiQALoadResult
from optirag.indexing import pinecone_lifecycle as pl
from optirag.optimization.trial_params import Stage1TrialParams


@dataclass
class _FakeSettings:
    artifacts_dir: Path
    index_force_fresh: bool = False
    index_upsert_max_retries: int = 2
    index_upsert_backoff_base_seconds: float = 0.001


class _FakeEmbedder:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(i + 1)] for i, _ in enumerate(texts)]


class _FakeRetriever:
    stores: dict[str, dict[str, dict[str, Any]]] = {}
    fail_once_on_id: str | None = None

    def __init__(self, *, index_host: str, namespace: str) -> None:
        self.host = index_host
        self.namespace = namespace
        _FakeRetriever.stores.setdefault(namespace, {})

    def delete_namespace(self) -> None:
        _FakeRetriever.stores[self.namespace] = {}

    def upsert_batch(
        self,
        vectors: list[list[float]],
        ids: list[str],
        metadata: list[dict[str, Any]],
    ) -> None:
        for _vec, vid, meta in zip(vectors, ids, metadata, strict=True):
            if _FakeRetriever.fail_once_on_id == vid:
                _FakeRetriever.fail_once_on_id = None
                raise RuntimeError("simulated transient failure")
            _FakeRetriever.stores[self.namespace][vid] = meta

    def namespace_vector_count(self) -> int | None:
        return len(_FakeRetriever.stores[self.namespace])


def _load_result() -> FiQALoadResult:
    return FiQALoadResult(
        corpus={"d1": "doc one", "d2": "doc two", "d3": "doc three"},
        queries={},
        qrels={},
        split="test",
    )


def _chunk_stub(corpus: dict[str, str], p: Stage1TrialParams) -> list[Any]:
    del corpus, p
    return [
        type("Chunk", (), {"beir_corpus_id": "d1", "text": "a", "chunk_index": 0})(),
        type("Chunk", (), {"beir_corpus_id": "d2", "text": "b", "chunk_index": 0})(),
        type("Chunk", (), {"beir_corpus_id": "d3", "text": "c", "chunk_index": 0})(),
    ]


def _configure(monkeypatch: Any, tmp_path: Path, *, force_fresh: bool = False, reset_store: bool = True) -> None:
    if reset_store:
        _FakeRetriever.stores = {}
        _FakeRetriever.fail_once_on_id = None
    s = _FakeSettings(artifacts_dir=tmp_path, index_force_fresh=force_fresh)
    monkeypatch.setattr(pl, "get_settings", lambda: s)
    monkeypatch.setattr(pl, "resolve_physical_index_host", lambda _d, _m: "fake-host")
    monkeypatch.setattr(pl, "GeminiEmbedder", _FakeEmbedder)
    monkeypatch.setattr(pl, "PineconeRetriever", _FakeRetriever)
    monkeypatch.setattr(pl, "chunk_corpus", _chunk_stub)
    monkeypatch.setattr(pl.time, "sleep", lambda _x: None)


def test_resume_replays_failed_batch(monkeypatch: Any, tmp_path: Path) -> None:
    _configure(monkeypatch, tmp_path)
    def _chunk_many(_corpus: dict[str, str], _p: Stage1TrialParams) -> list[Any]:
        return [
            type("Chunk", (), {"beir_corpus_id": f"d{i}", "text": f"t{i}", "chunk_index": 0})()
            for i in range(40)
        ]

    monkeypatch.setattr(pl, "chunk_corpus", _chunk_many)
    s = pl.get_settings()
    s.index_upsert_max_retries = 1
    _FakeRetriever.fail_once_on_id = "d33:0"
    p = Stage1TrialParams()
    loaded = _load_result()
    with pytest.raises(RuntimeError):
        pl.ensure_corpus_indexed(loaded, p, corpus_version="cv1")

    fp = pl.stage1_index_fingerprint("cv1", p)
    progress_path = tmp_path / "index_cache" / fp / "progress.json"
    data = json.loads(progress_path.read_text(encoding="utf-8"))
    assert data["next_offset"] == 32

    s.index_upsert_max_retries = 2
    ic = pl.ensure_corpus_indexed(loaded, p, corpus_version="cv1")
    assert ic.num_vectors == 40
    assert len(_FakeRetriever.stores[ic.namespace]) == 40


def test_force_fresh_clears_namespace(monkeypatch: Any, tmp_path: Path) -> None:
    _configure(monkeypatch, tmp_path)
    p = Stage1TrialParams()
    loaded = _load_result()
    pl.ensure_corpus_indexed(loaded, p, corpus_version="cv2")
    ns = pl.pinecone_namespace_id(pl.stage1_index_fingerprint("cv2", p))
    _FakeRetriever.stores[ns]["stale:0"] = {"stale": True}

    _configure(monkeypatch, tmp_path, force_fresh=True, reset_store=False)
    pl.ensure_corpus_indexed(loaded, p, corpus_version="cv2")
    assert "stale:0" not in _FakeRetriever.stores[ns]
    assert len(_FakeRetriever.stores[ns]) == 3


def test_incompatible_progress_is_ignored(monkeypatch: Any, tmp_path: Path) -> None:
    _configure(monkeypatch, tmp_path)
    p = Stage1TrialParams()
    fp = pl.stage1_index_fingerprint("cv3", p)
    cdir = tmp_path / "index_cache" / fp
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "progress.json").write_text(
        json.dumps(
            {
                "fingerprint": fp,
                "namespace": "wrong-ns",
                "host": "fake-host",
                "total_chunks": 999,
                "batch_size": 32,
                "next_offset": 2,
                "updated_at": "now",
            }
        ),
        encoding="utf-8",
    )

    ic = pl.ensure_corpus_indexed(_load_result(), p, corpus_version="cv3")
    assert ic.num_vectors == 3


def test_manifest_not_written_when_reconciliation_fails(monkeypatch: Any, tmp_path: Path) -> None:
    class _MismatchRetriever(_FakeRetriever):
        def namespace_vector_count(self) -> int | None:
            return 2

    _configure(monkeypatch, tmp_path)
    monkeypatch.setattr(pl, "PineconeRetriever", _MismatchRetriever)
    p = Stage1TrialParams()
    with pytest.raises(RuntimeError):
        pl.ensure_corpus_indexed(_load_result(), p, corpus_version="cv4")

    fp = pl.stage1_index_fingerprint("cv4", p)
    man_path = tmp_path / "index_cache" / fp / "manifest.json"
    assert not man_path.exists()
