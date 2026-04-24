"""
Locked FiQA/BEIR evaluation protocol (plan §7).

- Qrels reference **parent** BEIR corpus line ids.
- Retrieved items are **chunks** with metadata `beir_corpus_id` = parent id.
- Ground-truth **contexts** for RAGAS: full passage text of each qrel-relevant doc.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


class QrelEvalProtocol(StrEnum):
    """How to align RAGAS rows with BEIR qrels."""

    # Use full qrel passage texts as reference contexts; retrieved = chunk texts
    PARENT_ID_REF_PASSAGE_TEXT = "parent_id_ref_passage_text"
    # Doc-level IR ablation: max score per parent in ranked list (not used in RAGAS directly)
    PARENT_ID_MAX_SCORE = "parent_id_max_score"


def build_ground_truth_contexts(
    qrel_doc_ids: set[str],
    corpus_by_id: Mapping[str, str],
) -> list[str]:
    """Return full BEIR line text for each relevant doc id (order stable for reproducibility)."""
    return [corpus_by_id[did] for did in sorted(qrel_doc_ids) if did in corpus_by_id]
