"""
Compatibility: (pinecone_metric, l2_normalize) pairs.

Illegal combinations are pruned in Optuna search_space.
"""

from __future__ import annotations

# Dot product with *non*-normalized vectors is valid but geometry differs; we allow
# (cosine, True), (cosine, False) is odd for index — many implementations normalize for cosine
# Simplified allowed set: prefer normalized for cosine and dotproduct; euclidean may be either
ALLOWED_COMBINATIONS: set[tuple[str, bool]] = {
    ("cosine", True),
    ("cosine", False),
    ("dotproduct", True),
    ("dotproduct", False),
    ("euclidean", True),
    ("euclidean", False),
}


def is_allowed_pair(metric: str, l2_normalize: bool) -> bool:
    return (metric, l2_normalize) in ALLOWED_COMBINATIONS
