from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

if "pinecone" not in sys.modules:
    pinecone_stub = types.ModuleType("pinecone")
    pinecone_stub.Pinecone = object
    sys.modules["pinecone"] = pinecone_stub

from optirag.cli import index_cmd
from optirag.indexing.pinecone_lifecycle import IndexedCorpus
from optirag.optimization.trial_params import Stage1TrialParams


@dataclass
class _FakeSettings:
    data_dir: Path
    artifacts_dir: Path


class _FakeExperiment:
    data_split = "test"
    fiqa_max_docs = 2

    def resolved_stage1_params(self) -> Stage1TrialParams:
        return Stage1TrialParams(top_k=7)

    def resolved_corpus_version(self) -> str:
        return "fake-corpus-v1"


def test_index_build_cli_writes_last_manifest(monkeypatch: Any, tmp_path: Path) -> None:
    runner = CliRunner()
    fake_settings = _FakeSettings(data_dir=tmp_path / "data", artifacts_dir=tmp_path / "artifacts")
    fake_settings.data_dir.mkdir(parents=True, exist_ok=True)
    fake_loaded = object()
    calls: dict[str, Any] = {}

    def _fake_load_experiment(path: Path) -> _FakeExperiment:
        calls["experiment_path"] = str(path)
        return _FakeExperiment()

    def _fake_load_fiqa(data_path: Path, *, split: Any, max_docs: int | None = None) -> object:
        calls["data_path"] = str(data_path)
        calls["split"] = str(split)
        calls["max_docs"] = max_docs
        return fake_loaded

    def _fake_ensure(
        loaded: object,
        p: Stage1TrialParams,
        *,
        corpus_version: str,
        force_rebuild: bool = False,
    ) -> IndexedCorpus:
        calls["loaded"] = loaded
        calls["top_k"] = p.top_k
        calls["corpus_version"] = corpus_version
        calls["force_rebuild"] = force_rebuild
        return IndexedCorpus(
            host="fake-host",
            namespace="opt-fake",
            fingerprint="abc123",
            index_key="idx-key",
            from_cache=False,
            num_vectors=42,
        )

    monkeypatch.setattr(index_cmd, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(index_cmd, "load_experiment", _fake_load_experiment)
    monkeypatch.setattr(index_cmd, "load_fiqa", _fake_load_fiqa)
    monkeypatch.setattr(index_cmd, "ensure_corpus_indexed", _fake_ensure)

    exp_path = tmp_path / "exp.yaml"
    exp_path.write_text("name: fake\n", encoding="utf-8")
    result = runner.invoke(index_cmd.app, ["build", "--experiment", str(exp_path), "--force"])
    assert result.exit_code == 0, result.output

    out_path = fake_settings.artifacts_dir / "last_index_build.json"
    assert out_path.is_file()
    manifest = json.loads(out_path.read_text(encoding="utf-8"))
    assert manifest["num_vectors"] == 42
    assert manifest["namespace"] == "opt-fake"
    assert manifest["host"] == "fake-host"
    assert manifest["from_cache"] is False
    assert calls["loaded"] is fake_loaded
    assert calls["top_k"] == 7
    assert calls["corpus_version"] == "fake-corpus-v1"
    assert calls["force_rebuild"] is True
    assert Path(calls["data_path"]) == fake_settings.data_dir / "fiqa"
