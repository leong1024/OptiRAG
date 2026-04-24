from optirag.optimization.trial_params import Stage1TrialParams
from optirag.preprocessing.chunking import chunk_passage


def test_identity_chunk() -> None:
    p = Stage1TrialParams(chunk_strategy="identity_one_vec_per_line")
    ch = chunk_passage("d1", "short text", p)
    assert len(ch) == 1
    assert ch[0].beir_corpus_id == "d1"


def test_fixed_window() -> None:
    p = Stage1TrialParams(
        chunk_strategy="fixed_window",
        chunk_size=10,
        chunk_overlap=0,
        min_chunk_chars=0,
    )
    t = "a" * 25
    ch = chunk_passage("d1", t, p)
    assert len(ch) >= 2
