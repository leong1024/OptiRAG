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
from optirag.eval.runner import write_report
from optirag.optimization.objective import ObjectiveContext, run_single_config_eval
from optirag.optimization.trial_params import Stage1TrialParams
from optirag.rag.pipeline import run_rag_query

app = typer.Typer(no_args_is_help=True, help="Run RAG + RAGAS eval")


@app.command("run")
def run_eval(
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
    report = run_single_config_eval(p, ctx)
    out = s.artifacts_dir / "reports" / f"eval_{exp.name}.json"
    write_report(report, out)
    typer.echo(f"mean_scalar={report.mean_scalar:.4f} wrote {out}")
