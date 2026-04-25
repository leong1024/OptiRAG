from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
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


@dataclass(frozen=True, slots=True)
class ProgressState:
    fingerprint: str
    namespace: str
    host: str
    total_chunks: int
    batch_size: int
    next_offset: int
    updated_at: str


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


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _progress_from_dict(data: dict[str, Any]) -> ProgressState | None:
    try:
        return ProgressState(
            fingerprint=str(data["fingerprint"]),
            namespace=str(data["namespace"]),
            host=str(data["host"]),
            total_chunks=int(data["total_chunks"]),
            batch_size=int(data["batch_size"]),
            next_offset=int(data["next_offset"]),
            updated_at=str(data.get("updated_at", "")),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _save_progress(path: Path, prog: ProgressState) -> None:
    _atomic_write_json(
        path,
        {
            "fingerprint": prog.fingerprint,
            "namespace": prog.namespace,
            "host": prog.host,
            "total_chunks": prog.total_chunks,
            "batch_size": prog.batch_size,
            "next_offset": prog.next_offset,
            "updated_at": prog.updated_at,
        },
    )


def _retry_upsert_batch(
    retriever: PineconeRetriever,
    vectors: list[list[float]],
    ids: list[str],
    metadata: list[dict[str, Any]],
) -> None:
    s = get_settings()
    max_retries = max(1, int(s.index_upsert_max_retries))
    backoff_base = max(0.05, float(s.index_upsert_backoff_base_seconds))
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            retriever.upsert_batch(vectors, ids, metadata)
            return
        except Exception as e:  # noqa: BLE001
            last_error = e
            if attempt == max_retries:
                break
            wait_s = backoff_base * (2 ** (attempt - 1))
            logger.warning(
                "Pinecone upsert batch retry %d/%d in %.2fs: %s",
                attempt,
                max_retries,
                wait_s,
                e,
            )
            time.sleep(wait_s)
    assert last_error is not None
    raise RuntimeError(f"Pinecone upsert failed after retries: {last_error}") from last_error


@contextmanager
def _index_lock(lock_path: Path) -> Any:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        msg = f"Index build already running for this fingerprint (lock exists: {lock_path})"
        raise RuntimeError(msg) from exc
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove lock file: %s", lock_path)


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
    progress_path = cache_dir / "progress.json"
    lock_path = cache_dir / ".build.lock"
    dim = p.embedding_dim()
    host = resolve_physical_index_host(dim, p.pinecone_metric)
    force_fresh = bool(force_rebuild or s.index_force_fresh)

    if not force_fresh and man_path.is_file():
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
    with _index_lock(lock_path):
        mode = "force-fresh" if force_fresh else "resuming"
        trigger = "cli_force" if force_rebuild else ("env_force_fresh" if s.index_force_fresh else "none")
        logger.info("Index build mode=%s trigger=%s fingerprint=%s namespace=%s", mode, trigger, fp, ns)

        retriever = PineconeRetriever(index_host=host, namespace=ns)
        if force_fresh:
            retriever.delete_namespace()
            progress_path.unlink(missing_ok=True)
            man_path.unlink(missing_ok=True)

        embedder = GeminiEmbedder(
            p.embedding_model,
            output_dimensionality=p.output_dim_override,
            l2_normalize=p.l2_normalize,
        )
        chunks = chunk_corpus(loaded.corpus, p)
        texts = [c.text for c in chunks]
        ids = [f"{c.beir_corpus_id}:{c.chunk_index}" for c in chunks]
        total_chunks = len(texts)
        batch = 32

        base_meta = [
            {
                "beir_corpus_id": c.beir_corpus_id,
                "text": c.text,
                "chunk_index": c.chunk_index,
                "fingerprint": fp,
                "embedding_model": p.embedding_model,
                "embedding_dim": dim,
                "l2_normalize": p.l2_normalize,
                "corpus_version": corpus_version,
            }
            for c in chunks
        ]

        start_offset = 0
        if not force_fresh and progress_path.is_file():
            data = _load_json(progress_path)
            prog = _progress_from_dict(data)
            if prog is None:
                logger.warning("Malformed progress file; restarting at offset 0 (%s)", progress_path)
            elif (
                prog.fingerprint == fp
                and prog.host == host
                and prog.namespace == ns
                and prog.total_chunks == total_chunks
                and prog.batch_size == batch
            ):
                start_offset = max(0, min(prog.next_offset, total_chunks))
                logger.info("Resuming index build from offset=%d/%d", start_offset, total_chunks)
            else:
                logger.warning("Ignoring incompatible progress file for fingerprint=%s", fp)

        total_batches = (total_chunks + batch - 1) // batch if total_chunks else 0
        start_batch = (start_offset // batch) + 1 if total_batches else 0
        for batch_idx, i in enumerate(range(start_offset, total_chunks, batch), start=start_batch):
            batch_texts = texts[i : i + batch]
            batch_ids = ids[i : i + batch]
            batch_meta = base_meta[i : i + batch]
            batch_vecs = embedder.embed_documents(batch_texts)
            _retry_upsert_batch(retriever, batch_vecs, batch_ids, batch_meta)
            next_offset = i + len(batch_texts)
            _save_progress(
                progress_path,
                ProgressState(
                    fingerprint=fp,
                    namespace=ns,
                    host=host,
                    total_chunks=total_chunks,
                    batch_size=batch,
                    next_offset=next_offset,
                    updated_at=_now_iso(),
                ),
            )
            if batch_idx == 1 or batch_idx == total_batches or batch_idx % 10 == 0:
                pct = (next_offset / total_chunks * 100.0) if total_chunks else 100.0
                logger.info(
                    "Indexing progress: %d/%d chunks (%.1f%%), batch %d/%d",
                    next_offset,
                    total_chunks,
                    pct,
                    batch_idx,
                    total_batches,
                )

        ns_count = retriever.namespace_vector_count()
        if ns_count is not None and ns_count != total_chunks:
            msg = (
                f"Namespace vector count mismatch for {ns}: expected={total_chunks}, "
                f"actual={ns_count}. Manifest not written."
            )
            raise RuntimeError(msg)

        manifest = {
            "fingerprint": fp,
            "trial_params_fingerprint": trial_params_fingerprint(p),
            "namespace": ns,
            "host": host,
            "index_key": idx_key,
            "num_vectors": len(ids),
            "corpus_version": corpus_version,
            "embedding_model": p.embedding_model,
            "embedding_dim": dim,
            "l2_normalize": p.l2_normalize,
            "updated_at": _now_iso(),
        }
        _atomic_write_json(man_path, manifest)
        progress_path.unlink(missing_ok=True)
        logger.info("Indexed corpus fp=%s vectors=%d namespace=%s", fp, len(ids), ns)
        return IndexedCorpus(
            host=host,
            namespace=ns,
            fingerprint=fp,
            index_key=idx_key,
            from_cache=False,
            num_vectors=len(ids),
        )
