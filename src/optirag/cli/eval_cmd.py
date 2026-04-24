from __future__ import annotations

from pathlib import Path

import typer

from optirag.config.experiment import load_experiment
from optirag.config.settings import get_settings
from optirag.data.beir_fiqa import load_fiqa
from optirag.domain.types import DataSplit
from optirag.eval.runner import write_report
from optirag.optimization.objective import ObjectiveContext, run_single_config_eval

app = typer.Typer(no_args_is_help=True, help="Run RAG + RAGAS eval")


@app.command("run")
def run_eval(
    experiment: Path = typer.Option(Path("experiments/fiqa_stage1.yaml"), exists=True),
) -> None:
    exp = load_experiment(experiment)
    s = get_settings()
    loaded = load_fiqa(s.data_dir / "fiqa", split=DataSplit(exp.data_split))
    p = exp.resolved_stage1_params()
    ctx = ObjectiveContext(
        data=loaded,
        experiment=exp,
        corpus_version=exp.name,
    )
    report = run_single_config_eval(p, ctx)
    out = s.artifacts_dir / "reports" / f"eval_{exp.name}.json"
    write_report(report, out)
    typer.echo(f"mean_scalar={report.mean_scalar:.4f} wrote {out}")
