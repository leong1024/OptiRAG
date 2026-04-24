from __future__ import annotations

from typing import TYPE_CHECKING, Any

from optirag.eval.metrics import ScalarWeights, composite_scalar
from optirag.eval.qrel_protocol import QrelEvalProtocol, build_ground_truth_contexts

if TYPE_CHECKING:
    from optirag.eval.runner import EvalReport, run_rag_eval

__all__ = [
    "ScalarWeights",
    "composite_scalar",
    "QrelEvalProtocol",
    "build_ground_truth_contexts",
    "EvalReport",
    "run_rag_eval",
]


def __getattr__(name: str) -> Any:
    if name in ("EvalReport", "run_rag_eval"):
        from optirag.eval import runner

        if name == "EvalReport":
            return runner.EvalReport
        return runner.run_rag_eval
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
