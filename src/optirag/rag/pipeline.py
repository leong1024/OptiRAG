from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from optirag.domain.types import RAGResult, RetrievedPassage
from optirag.optimization.trial_params import Stage1TrialParams
from optirag.rag.frozen_stage1 import FrozenStage1Prompts, identity_retrieval_query

if TYPE_CHECKING:
    from optirag.adapters.gemini.chat import GeminiLlm
    from optirag.adapters.gemini.embedder import GeminiEmbedder
    from optirag.adapters.pinecone.store import PineconeRetriever


def _apply_post_retrieval(
    raw: list[tuple[str, float, dict[str, Any]]],
    p: Stage1TrialParams,
) -> list[RetrievedPassage]:
    """min_similarity filter, optional dedup by parent, cap chunks per parent."""
    rows: list[RetrievedPassage] = []
    for vid, score, meta in raw:
        if p.min_similarity is not None and score < p.min_similarity:
            continue
        pid = str(meta.get("beir_corpus_id", ""))
        text = str(meta.get("text", ""))
        rows.append(
            RetrievedPassage(
                beir_corpus_id=pid,
                text=text,
                score=score,
                vector_id=vid,
            )
        )
    rows.sort(key=lambda x: -x.score)
    if p.parent_dedup_policy == "keep_highest_score":
        seen: set[str] = set()
        kept: list[RetrievedPassage] = []
        for r in rows:
            if r.beir_corpus_id in seen:
                continue
            seen.add(r.beir_corpus_id)
            kept.append(r)
        rows = kept
    per_parent: dict[str, list[RetrievedPassage]] = defaultdict(list)
    for r in rows:
        per_parent[r.beir_corpus_id].append(r)
    limited: list[RetrievedPassage] = []
    for _pid, lst in per_parent.items():
        lst = sorted(lst, key=lambda x: -x.score)[: p.max_chunks_per_beir_id]
        limited.extend(lst)
    limited.sort(key=lambda x: -x.score)
    return limited


def _build_context(ranked: list[RetrievedPassage], budget: int) -> tuple[str, list[str]]:
    parts: list[str] = []
    used = 0
    parent_ids: list[str] = []
    for r in ranked:
        block = r.text
        if used + len(block) + 2 > budget and parts:
            break
        parts.append(block)
        parent_ids.append(r.beir_corpus_id)
        used += len(block) + 2
    return "\n\n".join(parts), parent_ids


def run_rag_query(
    *,
    query_id: str,
    user_query: str,
    trial: Stage1TrialParams,
    embedder: GeminiEmbedder,
    retriever: PineconeRetriever,
    llm: GeminiLlm,
    prompts: FrozenStage1Prompts | None = None,
    use_query_llm: bool = False,
) -> RAGResult:
    """
    RAG: identity or frozen rewrite -> embed -> retrieve -> post-process -> answer.
    For Stage 1, pass use_query_llm=False and use identity_retrieval_query.
    """
    prompts = prompts or FrozenStage1Prompts()
    if use_query_llm:
        rq = llm.complete(prompts.query_system, user_query, temperature=0.0, max_tokens=512)
    else:
        rq = identity_retrieval_query(user_query)
    qv = embedder.embed_query(rq)
    raw = retriever.query(qv, top_k=trial.top_k)
    ranked = _apply_post_retrieval(raw, trial)
    context, pids = _build_context(ranked, trial.context_char_budget)
    user_block = f"Context:\n{context}\n\nQuestion:\n{user_query}"
    answer = llm.complete(
        prompts.answer_system,
        user_block,
        temperature=0.0,
        max_tokens=1024,
    )
    return RAGResult(
        query_id=query_id,
        user_query=user_query,
        retrieval_query=rq,
        retrieved=ranked,
        answer=answer,
        contexts_for_eval=[r.text for r in ranked],
        parent_ids_in_context=pids,
    )
