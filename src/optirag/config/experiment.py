from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from optirag.domain.types import DataSplit
from optirag.eval.metrics import ScalarWeights
from optirag.optimization.trial_params import Stage1TrialParams


class RagasConfig(BaseModel):
    """Metrics and scalar weights for Optuna / reports."""

    metric_names: list[str] = Field(
        default_factory=lambda: [
            "context_precision",
            "context_recall",
            "faithfulness",
            "answer_relevancy",
        ]
    )
    weights: ScalarWeights = Field(default_factory=ScalarWeights)
    query_subset: int | None = Field(
        default=None,
        description="If set, cap number of eval queries (for cost).",
    )
    seed: int = 42


class OptunaConfig(BaseModel):
    n_trials: int = 20
    direction: str = "maximize"
    two_phase: bool = False
    study_name: str = "optirag-stage1"
    storage: str | None = Field(
        default=None,
        description=(
            "Optuna storage URL (for resume/checkpoint), e.g. "
            "'sqlite:///artifacts/optuna/stage1.db'. "
            "If omitted, CLI defaults to a persistent sqlite DB in artifacts/optuna."
        ),
    )
    tune_index_hyperparams: bool = Field(
        default=False,
        description=(
            "If true, Optuna varies chunk/embedding/metric (separate index per trial). "
            "If false, only retrieval/app-layer params (fixed index)."
        ),
    )


class ExperimentConfig(BaseModel):
    """Loaded from experiments/*.yaml."""

    name: str = "fiqa_stage1"
    dataset: str = "fiqa"
    data_split: DataSplit = DataSplit.TEST
    qrel_eval_protocol: str = "parent_id_ref_passage_text"
    ragas: RagasConfig = Field(default_factory=RagasConfig)
    optuna: OptunaConfig = Field(default_factory=OptunaConfig)
    extra: dict[str, Any] = Field(default_factory=dict)
    stage1_base: dict[str, Any] | None = Field(
        default=None,
        description="Overrides merged onto `Stage1TrialParams()` for index build and retrieval-only base.",
    )

    def resolved_stage1_params(self) -> Stage1TrialParams:
        """Defaults + optional `stage1_base` from YAML."""
        merged: dict[str, Any] = asdict(Stage1TrialParams())
        if self.stage1_base:
            merged.update(self.stage1_base)
        return Stage1TrialParams.from_dict(merged)

    @classmethod
    def load(cls, path: Path) -> ExperimentConfig:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            msg = "Experiment YAML must be a mapping"
            raise ValueError(msg)
        return cls.model_validate(raw)


def load_experiment(path: Path | str) -> ExperimentConfig:
    return ExperimentConfig.load(Path(path))
