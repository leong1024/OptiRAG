from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Embed query strings and document strings with a given model id."""

    model_id: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text."""

    def embed_query(self, text: str) -> list[float]:
        ...
