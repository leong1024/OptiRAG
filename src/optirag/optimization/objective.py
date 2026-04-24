from __future__ import annotations

import logging
from dataclasses import dataclass

from optirag.adapters.gemini.chat import GeminiLlm
from optirag.adapters.gemini.embedder import GeminiEmbedder
from optirag.adapters.pinecone.store import PineconeRetriever
from optirag.config.experiment import ExperimentConfig
from optirag.data.beir_fiqa import FiQALoadResult
from optirag.eval.dataset import rows_from_rag_results
from optirag.eval.runner import EvalReport, run_rag_eval
from optirag.indexing.pinecone_lifecycle import ensure_corpus_indexed
from optirag.optimization.trial_params import Stage1TrialParams
from optirag.rag.pipeline import run_rag_query

logger = logging.getLogger(__name__)


@dataclass
class ObjectiveContext:
    data: FiQALoadResult
    experiment: ExperimentConfig
    corpus_version: str
    max_queries: int | None = None


def run_single_config_eval(
    trial_params: Stage1TrialParams,
    ctx: ObjectiveContext,
) -> EvalReport:
    """Run RAG over (subset of) queries and return RAGAS report."""
    ic = ensure_corpus_indexed(
        ctx.data,
        trial_params,
        corpus_version=ctx.corpus_version,
    )
    if ic.from_cache:
        logger.debug("Using cached index build fp=%s", ic.fingerprint)
    embedder = GeminiEmbedder(
        trial_params.embedding_model,
        output_dimensionality=trial_params.output_dim_override,
        l2_normalize=trial_params.l2_normalize,
    )
    retriever = PineconeRetriever(index_host=ic.host, namespace=ic.namespace)
    llm = GeminiLlm()

    queries = list(ctx.data.queries.items())
    if ctx.experiment.ragas.query_subset:
        queries = queries[: ctx.experiment.ragas.query_subset]
    if ctx.max_queries:
        queries = queries[: ctx.max_queries]
    results = []
    for qid, qtext in queries:
        r = run_rag_query(
            query_id=qid,
            user_query=qtext,
            trial=trial_params,
            embedder=embedder,
            retriever=retriever,
            llm=llm,
            use_query_llm=False,
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
