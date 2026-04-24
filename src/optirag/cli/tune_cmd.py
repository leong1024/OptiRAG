from __future__ import annotations

from functools import partial
from pathlib import Path

import typer

from optirag.adapters.gemini.chat import GeminiLlm
from optirag.adapters.gemini.embedder import GeminiEmbedder
from optirag.adapters.pinecone.store import PineconeRetriever
from optirag.config.experiment import load_experiment
from optirag.config.settings import get_settings
from optirag.data.beir_fiqa import load_fiqa
from optirag.domain.types import DataSplit
from optirag.optimization.objective import ObjectiveContext
from optirag.optimization.study import run_optuna_stage1
from optirag.optimization.trial_params import Stage1TrialParams
from optirag.rag.pipeline import run_rag_query

app = typer.Typer(no_args_is_help=True, help="Optuna tuning")


@app.command("stage1")
def tune_stage1(
    experiment: Path = typer.Option(Path("experiments/fiqa_stage1.yaml"), exists=True),
) -> None:
    exp = load_experiment(experiment)
    s = get_settings()
    loaded = load_fiqa(s.data_dir / "fiqa", split=DataSplit(exp.data_split))
    p = Stage1TrialParams()
    embedder = GeminiEmbedder(
        p.embedding_model,
        output_dimensionality=p.output_dim_override,
        l2_normalize=p.l2_normalize,
    )
    retriever = PineconeRetriever()
    llm = GeminiLlm()
    run = partial(
        run_rag_query,
        embedder=embedder,
        retriever=retriever,
        llm=llm,
        use_query_llm=False,
    )
    ctx = ObjectiveContext(
        data=loaded,
        experiment=exp,
        corpus_version=exp.name,
        run_rag_fn=lambda **kw: run(**kw),
    )
    study = run_optuna_stage1(
        ctx,
        exp,
        n_trials=exp.optuna.n_trials,
        study_name=exp.optuna.study_name,
        artifacts_dir=s.artifacts_dir / "optuna",
    )
    typer.echo(f"best_value={study.best_value} best_params={study.best_params}")


@app.command("stage2")
def tune_stage2_stub() -> None:
    typer.echo("Use: optirag prompt-opt run  (requires pip install optirag[agents])")
