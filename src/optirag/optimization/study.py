from __future__ import annotations

import json
import logging
from pathlib import Path

import optuna
from optuna.trial import Trial

from optirag.config.experiment import ExperimentConfig
from optirag.optimization.objective import ObjectiveContext, run_single_config_eval
from optirag.optimization.search_space import suggest_retrieval_only, suggest_stage1_params
from optirag.optimization.trial_params import trial_params_fingerprint

logger = logging.getLogger(__name__)


def run_optuna_stage1(
    ctx: ObjectiveContext,
    experiment: ExperimentConfig,
    *,
    n_trials: int,
    storage: str | None = None,
    study_name: str = "optirag-s1",
    artifacts_dir: Path | None = None,
) -> optuna.Study:
    base = experiment.resolved_stage1_params()

    def objective(trial: Trial) -> float:
        if experiment.optuna.tune_index_hyperparams:
            p = suggest_stage1_params(trial, two_phase=experiment.optuna.two_phase)
        else:
            p = suggest_retrieval_only(trial, base=base)
        key = trial_params_fingerprint(p)
        trial.set_user_attr("param_fingerprint", key)
        report = run_single_config_eval(p, ctx)
        trial.set_user_attr("mean_scalar", report.mean_scalar)
        trial.set_user_attr("ragas", report.ragas_scores)
        if artifacts_dir:
            out = artifacts_dir / f"trial_{trial.number}_{key}.json"
            out.write_text(
                json.dumps(
                    {
                        "params": p.to_json_dict(),
                        "ragas": report.ragas_scores,
                        "mean_scalar": report.mean_scalar,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        return report.mean_scalar

    study = optuna.create_study(
        study_name=study_name,
        direction=experiment.optuna.direction,
        storage=storage,
        load_if_exists=bool(storage),
    )
    study.optimize(objective, n_trials=n_trials)
    return study
