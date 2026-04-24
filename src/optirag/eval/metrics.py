"""RAGAS scalar objective (weights configurable via experiment YAML)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class ScalarWeights(BaseModel):
    context_precision: float = Field(default=0.25, ge=0.0)
    context_recall: float = Field(default=0.25, ge=0.0)
    faithfulness: float = Field(default=0.25, ge=0.0)
    answer_relevancy: float = Field(default=0.25, ge=0.0)


def composite_scalar(
    ragas_row_scores: Mapping[str, float] | None,
    *,
    weights: ScalarWeights,
) -> float:
    """
    Build a single maximization objective from RAGAS metric columns.
    NaN or missing keys contribute 0 for that term (after renormalization).
    """
    if not ragas_row_scores:
        return 0.0
    keys = [
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_relevancy",
    ]
    w = np.array(
        [
            weights.context_precision,
            weights.context_recall,
            weights.faithfulness,
            weights.answer_relevancy,
        ],
        dtype=np.float64,
    )
    vals: list[float] = []
    for k in keys:
        v = ragas_row_scores.get(k)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            vals.append(0.0)
        else:
            vals.append(float(v))
    x = np.array(vals, dtype=np.float64)
    # If all zero weights, return mean of available
    if w.sum() <= 0:
        return float(np.mean(x)) if len(x) else 0.0
    wn = w / w.sum()
    return float((x * wn).sum())


def mean_composite_on_dataset(
    per_row: list[Mapping[str, Any]],
    *,
    weights: ScalarWeights,
) -> float:
    if not per_row:
        return 0.0
    scores: list[float] = []
    keys = ("context_precision", "context_recall", "faithfulness", "answer_relevancy")
    for row in per_row:
        d = {k: float(row[k]) for k in row if k in keys and row[k] is not None}
        s = composite_scalar(d, weights=weights)
        scores.append(s)
    return float(np.mean(scores)) if scores else 0.0
