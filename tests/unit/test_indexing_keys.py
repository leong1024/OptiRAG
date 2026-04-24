from optirag.optimization.trial_params import (
    Stage1TrialParams,
    index_cache_key,
    index_cache_key_from_params,
    pinecone_namespace_id,
    stage1_index_fingerprint,
)


def test_index_cache_key_includes_min_chunk_chars() -> None:
    a = index_cache_key(
        corpus_version="c",
        chunk_strategy="identity_one_vec_per_line",
        chunk_size=1024,
        chunk_overlap=0,
        min_chunk_chars=0,
        cleaning_mode="none",
        embedding_model="gemini-embedding-001",
        output_dim=3072,
        pinecone_metric="cosine",
        l2_normalize=True,
    )
    b = index_cache_key(
        corpus_version="c",
        chunk_strategy="identity_one_vec_per_line",
        chunk_size=1024,
        chunk_overlap=0,
        min_chunk_chars=100,
        cleaning_mode="none",
        embedding_model="gemini-embedding-001",
        output_dim=3072,
        pinecone_metric="cosine",
        l2_normalize=True,
    )
    assert a != b


def test_stage1_index_fingerprint_changes_with_min_chunk() -> None:
    p0 = Stage1TrialParams()
    p1 = Stage1TrialParams(min_chunk_chars=100)
    assert stage1_index_fingerprint("x", p0) != stage1_index_fingerprint("x", p1)


def test_pinecone_namespace_id() -> None:
    assert pinecone_namespace_id("abc12") == "opt-abc12"


def test_experiment_resolved_stage1_params() -> None:
    from optirag.config.experiment import ExperimentConfig

    e = ExperimentConfig(name="t", stage1_base={"top_k": 5})
    p = e.resolved_stage1_params()
    assert p.top_k == 5
    e2 = ExperimentConfig(name="t2")
    assert e2.resolved_stage1_params().top_k == 10


def test_index_cache_key_from_params_matches_manual() -> None:
    p = Stage1TrialParams(
        min_chunk_chars=50,
    )
    k1 = index_cache_key_from_params("cv1", p)
    k2 = index_cache_key(
        corpus_version="cv1",
        chunk_strategy=p.chunk_strategy,
        chunk_size=p.chunk_size,
        chunk_overlap=p.chunk_overlap,
        min_chunk_chars=p.min_chunk_chars,
        cleaning_mode=p.cleaning_mode,
        embedding_model=p.embedding_model,
        output_dim=p.embedding_dim(),
        pinecone_metric=p.pinecone_metric,
        l2_normalize=p.l2_normalize,
    )
    assert k1 == k2
