from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Retriever(Protocol):
    def upsert(
        self,
        vectors: list[list[float]],
        ids: list[str],
        metadata: list[dict[str, Any]],
    ) -> None: ...

    def query(
        self,
        vector: list[float],
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Return list of (id, score, metadata)."""
