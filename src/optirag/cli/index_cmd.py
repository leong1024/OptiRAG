from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from optirag.adapters.gemini.embedder import GeminiEmbedder
from optirag.adapters.pinecone.store import PineconeRetriever
from optirag.config.experiment import load_experiment
from optirag.config.settings import get_settings
from optirag.data.beir_fiqa import load_fiqa
from optirag.domain.types import DataSplit
from optirag.optimization.trial_params import (
    Stage1TrialParams,
    index_cache_key,
    trial_params_fingerprint,
)
from optirag.preprocessing.chunking import chunk_corpus

app = typer.Typer(no_args_is_help=True, help="Build Pinecone index")

logger = logging.getLogger(__name__)


@app.command("build")
def build(
    experiment: Path = typer.Option(
        Path("experiments/fiqa_stage1.yaml"),
        exists=True,
        help="Experiment YAML",
    ),
) -> None:
    """Chunk corpus, embed, upsert. Requires PINECONE_INDEX_HOST and a Pinecone index with matching dimension."""
    exp = load_experiment(experiment)
    s = get_settings()
    data_path = s.data_dir / "fiqa"
    loaded = load_fiqa(data_path, split=DataSplit(exp.data_split))
    p = Stage1TrialParams()
    embedder = GeminiEmbedder(
        p.embedding_model,
        output_dimensionality=p.output_dim_override,
        l2_normalize=p.l2_normalize,
    )
    dim = p.embedding_dim()
    chunks = chunk_corpus(loaded.corpus, p)
    texts = [c.text for c in chunks]
    metas = [
        {
            "beir_corpus_id": c.beir_corpus_id,
            "text": c.text,
            "chunk_index": c.chunk_index,
        }
        for c in chunks
    ]
    ids = [f"{c.beir_corpus_id}:{c.chunk_index}" for c in chunks]
    vecs: list[list[float]] = []
    batch = 32
    for i in range(0, len(texts), batch):
        vecs.extend(embedder.embed_documents(texts[i : i + batch]))
    retriever = PineconeRetriever()
    retriever.upsert(vecs, ids, metas)
    manifest = {
        "num_vectors": len(ids),
        "fingerprint": trial_params_fingerprint(p),
        "index_key": index_cache_key(
            corpus_version=exp.name,
            chunk_strategy=p.chunk_strategy,
            chunk_size=p.chunk_size,
            chunk_overlap=p.chunk_overlap,
            cleaning_mode=p.cleaning_mode,
            embedding_model=p.embedding_model,
            output_dim=dim,
            pinecone_metric=p.pinecone_metric,
            l2_normalize=p.l2_normalize,
        ),
    }
    out = s.artifacts_dir / "last_index_build.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    typer.echo(f"Upserted {len(ids)} vectors. Wrote {out}")
