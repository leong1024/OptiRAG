from __future__ import annotations

from typing import Any

from pinecone import Pinecone

from optirag.config.settings import get_settings


class PineconeRetriever:
    """Thin wrapper: upsert + query with metadata."""

    def __init__(self, index_host: str | None = None, namespace: str = "default") -> None:
        s = get_settings()
        if not s.pinecone_api_key:
            msg = "PINECONE_API_KEY is not set"
            raise ValueError(msg)
        self._pc = Pinecone(api_key=s.pinecone_api_key)
        host = index_host or s.pinecone_index_host
        if not host:
            msg = "Set PINECONE_INDEX_HOST to your serverless index host"
            raise ValueError(msg)
        self._index = self._pc.Index(host=host)
        self._namespace = namespace

    def upsert(
        self,
        vectors: list[list[float]],
        ids: list[str],
        metadata: list[dict[str, Any]],
    ) -> None:
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            chunk_ids = ids[i : i + batch_size]
            chunk_vecs = vectors[i : i + batch_size]
            chunk_meta = metadata[i : i + batch_size]
            self._index.upsert(
                vectors=[
                    {"id": vid, "values": v, "metadata": m}
                    for vid, v, m in zip(chunk_ids, chunk_vecs, chunk_meta, strict=True)
                ],
                namespace=self._namespace,
            )

    def query(
        self,
        vector: list[float],
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        res = self._index.query(
            vector=vector,
            top_k=top_k,
            namespace=self._namespace,
            filter=metadata_filter,
            include_metadata=True,
        )
        matches = getattr(res, "matches", None) or []
        out: list[tuple[str, float, dict[str, Any]]] = []
        for m in matches:
            mid = getattr(m, "id", "")
            sc = float(getattr(m, "score", 0.0))
            meta = dict(getattr(m, "metadata", {}) or {})
            out.append((mid, sc, meta))
        return out
