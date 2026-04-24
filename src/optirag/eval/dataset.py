from __future__ import annotations

from typing import Any

from datasets import Dataset

from optirag.domain.types import RAGResult
from optirag.eval.qrel_protocol import build_ground_truth_contexts


def build_ragas_dataset(
    rows: list[dict[str, Any]],
) -> Dataset:
    """
    Expected keys per row: user_input, retrieved_contexts (list[str]), response,
    reference (ground truth answer if any), reference_contexts (list[str]) optional.
    """
    return Dataset.from_list(rows)


def rows_from_rag_results(
    results: list[RAGResult],
    *,
    corpus: dict[str, str],
    qrels: dict[str, dict[str, int]],
    include_ground_truth_answer: bool = False,
    ground_truth_answers: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Build RAGAS rows: user_input, retrieved_contexts, response, reference (ground-truth string)."""
    out: list[dict[str, Any]] = []
    for r in results:
        rel = qrels.get(r.query_id, {})
        gt_ids = {doc_id for doc_id, grade in rel.items() if grade > 0}
        ref_ctx = build_ground_truth_contexts(gt_ids, corpus)
        ref_str = "\n\n".join(ref_ctx) if ref_ctx else ""
        row: dict[str, Any] = {
            "user_input": r.user_query,
            "retrieved_contexts": r.contexts_for_eval,
            "response": r.answer,
            "reference": ref_str,
        }
        if include_ground_truth_answer and ground_truth_answers:
            row["reference"] = ground_truth_answers.get(r.query_id, ref_str)
        out.append(row)
    return out
