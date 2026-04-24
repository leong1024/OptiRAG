from optirag.config.metric_normalize import is_allowed_pair


def test_allowed_pairs() -> None:
    assert is_allowed_pair("cosine", True)
    assert is_allowed_pair("euclidean", False)
