from __future__ import annotations

import csv
import json
from pathlib import Path

import typer
import optuna

from optirag.config.experiment import load_experiment
from optirag.config.settings import get_settings
from optirag.data.beir_fiqa import load_fiqa
from optirag.domain.types import DataSplit
from optirag.optimization.objective import ObjectiveContext
from optirag.optimization.study import run_optuna_stage1

app = typer.Typer(no_args_is_help=True, help="Optuna tuning")


@app.command("stage1")
def tune_stage1(
    experiment: Path = typer.Option(Path("experiments/fiqa_stage1.yaml"), exists=True),
) -> None:
    exp = load_experiment(experiment)
    s = get_settings()
    optuna_dir = s.artifacts_dir / "optuna"
    optuna_dir.mkdir(parents=True, exist_ok=True)
    storage = exp.optuna.storage or f"sqlite:///{(optuna_dir / f'{exp.optuna.study_name}.db').as_posix()}"
    loaded = load_fiqa(
        s.data_dir / "fiqa",
        split=DataSplit(exp.data_split),
        max_docs=exp.fiqa_max_docs,
    )
    ctx = ObjectiveContext(
        data=loaded,
        experiment=exp,
        corpus_version=exp.resolved_corpus_version(),
    )
    study = run_optuna_stage1(
        ctx,
        exp,
        n_trials=exp.optuna.n_trials,
        storage=storage,
        study_name=exp.optuna.study_name,
        artifacts_dir=optuna_dir,
    )
    typer.echo(f"best_value={study.best_value} best_params={study.best_params}")


@app.command("stage2")
def tune_stage2_stub() -> None:
    typer.echo("Use: optirag prompt-opt run  (requires pip install optirag[agents])")


@app.command("export-csv")
def export_csv(
    experiment: Path = typer.Option(Path("experiments/fiqa_stage1.yaml"), exists=True),
    output: Path | None = typer.Option(
        None,
        help="Output CSV path. Defaults to artifacts/optuna/<study_name>_trials.csv",
    ),
) -> None:
    """Export all Optuna trials to CSV for plotting/analysis."""
    exp = load_experiment(experiment)
    s = get_settings()
    optuna_dir = s.artifacts_dir / "optuna"
    optuna_dir.mkdir(parents=True, exist_ok=True)

    storage = exp.optuna.storage or f"sqlite:///{(optuna_dir / f'{exp.optuna.study_name}.db').as_posix()}"
    out_path = output or (optuna_dir / f"{exp.optuna.study_name}_trials.csv")

    study = optuna.load_study(study_name=exp.optuna.study_name, storage=storage)
    trials = study.get_trials(deepcopy=False)

    param_keys = sorted({k for t in trials for k in t.params})
    attr_keys = sorted({k for t in trials for k in t.user_attrs})
    fieldnames = [
        "trial_number",
        "state",
        "value",
        "datetime_start",
        "datetime_complete",
        "duration_seconds",
        *[f"param_{k}" for k in param_keys],
        *[f"user_attr_{k}" for k in attr_keys],
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in trials:
            row: dict[str, str | int | float | None] = {
                "trial_number": t.number,
                "state": str(t.state),
                "value": t.value,
                "datetime_start": t.datetime_start.isoformat() if t.datetime_start else "",
                "datetime_complete": t.datetime_complete.isoformat() if t.datetime_complete else "",
                "duration_seconds": t.duration.total_seconds() if t.duration else "",
            }
            for k in param_keys:
                row[f"param_{k}"] = t.params.get(k)
            for k in attr_keys:
                v = t.user_attrs.get(k)
                row[f"user_attr_{k}"] = json.dumps(v, ensure_ascii=True) if isinstance(v, (dict, list)) else v
            writer.writerow(row)

    typer.echo(f"Exported {len(trials)} trials to {out_path}")
