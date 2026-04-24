from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from optirag.adapters.gemini.embedder import GeminiEmbedder
from optirag.adapters.pinecone.store import PineconeRetriever
from optirag.config.settings import get_settings, pinecone_registry_path
from optirag.data.beir_fiqa import FiQALoadResult
from optirag.optimization.trial_params import (
    Stage1TrialParams,
    index_cache_key_from_params,
    pinecone_namespace_id,
    stage1_index_fingerprint,
    trial_params_fingerprint,
)
from optirag.preprocessing.chunking import chunk_corpus

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IndexedCorpus:
    """Where vectors for a (corpus, index-defining) config live in Pinecone."""

    host: str
    namespace: str
    fingerprint: str
    index_key: str
    from_cache: bool
    num_vectors: int


def _registry_key(dim: int, metric: str) -> str:
    return f"{dim}:{metric}"


def _load_registry(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}


def _save_registry(path: Path, reg: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reg, indent=2, sort_keys=True), encoding="utf-8")


def _physical_index_name(prefix: str, dim: int, metric: str) -> str:
    h = hashlib.sha256(f"{dim}:{metric}".encode()).hexdigest()[:8]
    base = f"{prefix}-d{dim}-m{metric[:3]}-{h}"
    return base[:45]


def _index_names(pc: object) -> set[str]:
    li = pc.list_indexes()  # type: ignore[union-attr]
    names = getattr(li, "names", None)
    if names:
        return set(names)
    idxs = getattr(li, "indexes", None) or []
    return {getattr(x, "name", str(x)) for x in idxs}


def resolve_physical_index_host(dim: int, metric: str) -> str:
    """
    One Pinecone index per (dimension, metric). Returns index host URL.
    Uses registry file, auto-create, or first-time binding to PINECONE_INDEX_HOST.
    """
    s = get_settings()
    if not s.pinecone_api_key:
        msg = "PINECONE_API_KEY is not set"
        raise ValueError(msg)
    rpath = pinecone_registry_path()
    reg = _load_registry(rpath)
    k = _registry_key(dim, metric)
    if k in reg:
        return reg[k]

    from pinecone import Pinecone, ServerlessSpec  # type: ignore[import-untyped]

    pc = Pinecone(api_key=s.pinecone_api_key)

    if s.pinecone_auto_create:
        name = _physical_index_name(s.pinecone_index_prefix, dim, metric)
        names = _index_names(pc)
        if name not in names:
            logger.info("Creating Pinecone index %s dim=%s metric=%s", name, dim, metric)
            pc.create_index(
                name=name,
                dimension=dim,
                metric=metric,
                spec=ServerlessSpec(cloud=s.pinecone_cloud, region=s.pinecone_region),
            )
        for _ in range(90):
            desc = pc.describe_index(name)  # type: ignore[union-attr]
            st = getattr(desc, "status", None)
            ready = getattr(st, "ready", None) if st is not None else None
            if ready is True or (st is not None and str(getattr(st, "state", "")) == "Ready"):
                break
            time.sleep(2)
        else:
            msg = f"Pinecone index {name} did not become ready in time"
            raise RuntimeError(msg)
        final = pc.describe_index(name)  # type: ignore[union-attr]
        host = str(getattr(final, "host", "") or "")
        if not host:
            msg = f"Could not read host for index {name}"
            raise RuntimeError(msg)
        reg[k] = host
        _save_registry(rpath, reg)
        return host

    if s.pinecone_index_host and not reg:
        reg[k] = s.pinecone_index_host
        _save_registry(rpath, reg)
        logger.info(
            "Bound PINECONE_INDEX_HOST to registry key %s. Add more keys to %s for other dim/metric.",
            k,
            rpath,
        )
        return s.pinecone_index_host

    if s.pinecone_index_host and k not in reg:
        msg = (
            f"No registry entry for {k}. Set OPTIRAG_PINECONE_AUTO_CREATE=true, "
            f"or add {k!r} to {rpath} with the correct index host, "
            f"or start with an empty registry and set PINECONE_INDEX_HOST to bind the first (dim, metric) only."
        )
        raise ValueError(msg)

    msg = (
        f"Unknown Pinecone index for {k}. Configure PINECONE_INDEX_HOST + empty registry, "
        f"OPTIRAG_PINECONE_AUTO_CREATE, or {rpath}."
    )
    raise ValueError(msg)


def ensure_corpus_indexed(
    loaded: FiQALoadResult,
    p: Stage1TrialParams,
    *,
    corpus_version: str,
    force_rebuild: bool = False,
) -> IndexedCorpus:
    """
    Build or reuse chunked+embedded vectors in Pinecone (namespace = index fingerprint).
    Caches by stage1_index_fingerprint under artifacts/index_cache/<fp>/manifest.json.
    """
    s = get_settings()
    art = s.artifacts_dir
    fp = stage1_index_fingerprint(corpus_version, p)
    ns = pinecone_namespace_id(fp)
    idx_key = index_cache_key_from_params(corpus_version, p)
    cache_dir = art / "index_cache" / fp
    man_path = cache_dir / "manifest.json"
    dim = p.embedding_dim()
    host = resolve_physical_index_host(dim, p.pinecone_metric)

    if not force_rebuild and man_path.is_file():
        try:
            man: dict[str, Any] = json.loads(man_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            man = {}
        if man.get("fingerprint") == fp and str(man.get("host", "")) == host and man.get("namespace") == ns:
            logger.info("Index cache hit fingerprint=%s namespace=%s", fp, ns)
            nv = int(man.get("num_vectors", 0))
            return IndexedCorpus(
                host=host,
                namespace=ns,
                fingerprint=fp,
                index_key=idx_key,
                from_cache=True,
                num_vectors=nv,
            )

    cache_dir.mkdir(parents=True, exist_ok=True)
    embedder = GeminiEmbedder(
        p.embedding_model,
        output_dimensionality=p.output_dim_override,
        l2_normalize=p.l2_normalize,
    )
    chunks = chunk_corpus(loaded.corpus, p)
    texts = [c.text for c in chunks]
    metas = [
        {
            "beir_corpus_id": c.beir_corpus_id,
            "text": c.text,
            "chunk_index": c.chunk_index,
        }
        for c in chunks
    ]
    ids = [f"{c.beir_corpus_id}:{c.chunk_index}" for c in chunks]
    vecs: list[list[float]] = []
    batch = 32
    for i in range(0, len(texts), batch):
        vecs.extend(embedder.embed_documents(texts[i : i + batch]))
    retriever = PineconeRetriever(index_host=host, namespace=ns)
    retriever.upsert(vecs, ids, metas)
    manifest = {
        "fingerprint": fp,
        "trial_params_fingerprint": trial_params_fingerprint(p),
        "namespace": ns,
        "host": host,
        "index_key": idx_key,
        "num_vectors": len(ids),
        "corpus_version": corpus_version,
    }
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Indexed corpus fp=%s vectors=%d namespace=%s", fp, len(ids), ns)
    return IndexedCorpus(
        host=host,
        namespace=ns,
        fingerprint=fp,
        index_key=idx_key,
        from_cache=False,
        num_vectors=len(ids),
    )
