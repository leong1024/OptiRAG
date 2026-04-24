from typing import Protocol, runtime_checkable


@runtime_checkable
class LlmClient(Protocol):
    def complete(self, system: str, user: str, *, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        ...
