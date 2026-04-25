from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from optirag.config.experiment import load_experiment
from optirag.config.settings import get_settings
from optirag.data.beir_fiqa import load_fiqa
from optirag.domain.types import DataSplit
from optirag.indexing.pinecone_lifecycle import ensure_corpus_indexed
from optirag.optimization.trial_params import trial_params_fingerprint

app = typer.Typer(no_args_is_help=True, help="Build Pinecone index")

logger = logging.getLogger(__name__)


@app.command("build")
def build(
    experiment: Path = typer.Option(
        Path("experiments/fiqa_stage1.yaml"),
        exists=True,
        help="Experiment YAML",
    ),
    force: bool = typer.Option(False, help="Rebuild even if cache manifest exists"),
) -> None:
    """Chunk corpus, embed, upsert. Uses `stage1_base` from the experiment YAML."""
    exp = load_experiment(experiment)
    s = get_settings()
    data_path = s.data_dir / "fiqa"
    loaded = load_fiqa(
        data_path,
        split=DataSplit(exp.data_split),
        max_docs=exp.fiqa_max_docs,
    )
    p = exp.resolved_stage1_params()
    ic = ensure_corpus_indexed(
        loaded,
        p,
        corpus_version=exp.resolved_corpus_version(),
        force_rebuild=force,
    )
    manifest = {
        "num_vectors": ic.num_vectors,
        "fingerprint": ic.fingerprint,
        "trial_params_fingerprint": trial_params_fingerprint(p),
        "index_key": ic.index_key,
        "namespace": ic.namespace,
        "host": ic.host,
        "from_cache": ic.from_cache,
    }
    out = s.artifacts_dir / "last_index_build.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    typer.echo(f"Index ready namespace={ic.namespace} wrote {out}")
