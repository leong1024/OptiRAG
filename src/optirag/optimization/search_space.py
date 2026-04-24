from __future__ import annotations

import optuna
from optuna.trial import Trial

from optirag.config.metric_normalize import is_allowed_pair
from optirag.optimization.trial_params import Stage1TrialParams


def suggest_retrieval_only(trial: Trial, *, base: Stage1TrialParams) -> Stage1TrialParams:
    """
    Vary only knobs that do not require a different embedded index (top_k, post-filter, app-layer).
    Use when the Pinecone index was built once with `optirag index build`.
    """
    top_k = trial.suggest_int("top_k", 3, 40, step=1)
    min_sim_raw = trial.suggest_categorical("min_similarity", ["none", "0.35", "0.45", "0.55"])
    min_sim: float | None = None if min_sim_raw == "none" else float(min_sim_raw)
    max_per = trial.suggest_categorical("max_chunks_per_beir_id", [1, 2, 3, 5])
    ctx_budget = trial.suggest_categorical("context_char_budget", [4000, 8000, 12000, 20000])
    dedup = trial.suggest_categorical("parent_dedup_policy", ["off", "keep_highest_score"])
    return Stage1TrialParams(
        cleaning_mode=base.cleaning_mode,
        chunk_strategy=base.chunk_strategy,
        chunk_size=base.chunk_size,
        chunk_overlap=base.chunk_overlap,
        min_chunk_chars=base.min_chunk_chars,
        embedding_model=base.embedding_model,
        output_dim_override=base.output_dim_override,
        l2_normalize=base.l2_normalize,
        pinecone_metric=base.pinecone_metric,
        top_k=top_k,
        min_similarity=min_sim,
        max_chunks_per_beir_id=int(max_per),
        context_char_budget=int(ctx_budget),
        parent_dedup_policy=dedup,  # type: ignore[arg-type]
        rerank_enabled=base.rerank_enabled,
        rerank_m=base.rerank_m,
    )


def suggest_stage1_params(trial: Trial, *, two_phase: bool = False) -> Stage1TrialParams:
    """
    Optuna suggest for plan §4.1. Prunes invalid (chunk_overlap, chunk_size) and
    illegal (metric, l2_normalize) if compatibility matrix is tightened later.
    """
    cleaning = trial.suggest_categorical("cleaning_mode", ["none", "light_normalize"])
    strategy = trial.suggest_categorical(
        "chunk_strategy",
        ["identity_one_vec_per_line", "fixed_window", "sliding_window", "recursive"],
    )
    chunk_size = trial.suggest_int("chunk_size", 256, 2048, step=128)
    overlap = trial.suggest_int("chunk_overlap", 0, min(512, chunk_size - 1), step=32)
    if overlap >= chunk_size:
        raise optuna.TrialPruned("overlap >= chunk_size")
    min_c = trial.suggest_categorical("min_chunk_chars", [0, 50, 100])
    model = trial.suggest_categorical("embedding_model", ["gemini-embedding-001", "gemini-embedding-002"])
    l2 = trial.suggest_categorical("l2_normalize", [True, False])
    if two_phase:
        metric: str = "cosine"
    else:
        metric = trial.suggest_categorical("pinecone_metric", ["cosine", "dotproduct", "euclidean"])
    if not is_allowed_pair(metric, l2):
        raise optuna.TrialPruned(f"pruned (metric, l2_normalize)=({metric},{l2})")
    top_k = trial.suggest_int("top_k", 3, 40, step=1)
    min_sim_raw = trial.suggest_categorical("min_similarity", ["none", "0.35", "0.45", "0.55"])
    min_sim: float | None = None if min_sim_raw == "none" else float(min_sim_raw)
    max_per = trial.suggest_categorical("max_chunks_per_beir_id", [1, 2, 3, 5])
    ctx_budget = trial.suggest_categorical("context_char_budget", [4000, 8000, 12000, 20000])
    dedup = trial.suggest_categorical("parent_dedup_policy", ["off", "keep_highest_score"])
    rerank = trial.suggest_categorical("rerank_enabled", [False, True])
    rrm = 20
    if rerank:
        rrm = trial.suggest_categorical("rerank_m", [20, 50, 100])
    return Stage1TrialParams(
        cleaning_mode=cleaning,  # type: ignore[arg-type]
        chunk_strategy=strategy,  # type: ignore[arg-type]
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        min_chunk_chars=min_c,
        embedding_model=model,
        l2_normalize=l2,
        pinecone_metric=metric,  # type: ignore[arg-type]
        top_k=top_k,
        min_similarity=min_sim,
        max_chunks_per_beir_id=max_per,
        context_char_budget=ctx_budget,
        parent_dedup_policy=dedup,  # type: ignore[arg-type]
        rerank_enabled=rerank,
        rerank_m=rrm,
    )
