from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from optirag.config.experiment import ExperimentConfig
from optirag.data.beir_fiqa import FiQALoadResult
from optirag.eval.dataset import rows_from_rag_results
from optirag.eval.runner import EvalReport, run_rag_eval
from optirag.optimization.trial_params import Stage1TrialParams

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class ObjectiveContext:
    data: FiQALoadResult
    experiment: ExperimentConfig
    corpus_version: str
    run_rag_fn: Callable[..., object]  # returns RAGResult
    max_queries: int | None = None


def run_single_config_eval(
    trial_params: Stage1TrialParams,
    ctx: ObjectiveContext,
) -> EvalReport:
    """Run RAG over (subset of) queries and return RAGAS report."""
    queries = list(ctx.data.queries.items())
    if ctx.experiment.ragas.query_subset:
        queries = queries[: ctx.experiment.ragas.query_subset]
    if ctx.max_queries:
        queries = queries[: ctx.max_queries]
    results = []
    for qid, qtext in queries:
        r = ctx.run_rag_fn(
            query_id=qid,
            user_query=qtext,
            trial=trial_params,
        )
        results.append(r)
    rows = rows_from_rag_results(
        results,
        corpus=ctx.data.corpus,
        qrels=ctx.data.qrels,
    )
    from datasets import Dataset

    ds = Dataset.from_list(rows)
    return run_rag_eval(ds, exp=ctx.experiment)
