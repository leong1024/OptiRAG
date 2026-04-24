import numpy as np

from optirag.eval.metrics import ScalarWeights, composite_scalar


def test_composite_scalar_weights() -> None:
    w = ScalarWeights(
        context_precision=1.0,
        context_recall=0.0,
        faithfulness=0.0,
        answer_relevancy=0.0,
    )
    s = composite_scalar(
        {
            "context_precision": 0.5,
            "context_recall": 0.2,
        },
        weights=w,
    )
    assert abs(s - 0.5) < 1e-6


def test_composite_ignores_nan() -> None:
    w = ScalarWeights()
    s = composite_scalar(
        {
            "context_precision": 1.0,
            "context_recall": float("nan"),
        },
        weights=w,
    )
    assert not np.isnan(s)
