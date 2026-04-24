from __future__ import annotations

import time

from google import genai

from optirag.config.settings import get_settings
from optirag.domain.errors import ConfigError, RetriableAPIError


class GeminiLlm:
    """Native Google Genai generate_content for answer generation (and optional query rewrite)."""

    def __init__(self, model_name: str | None = None, *, max_retries: int = 3) -> None:
        s = get_settings()
        if not s.gemini_api_key:
            msg = "GEMINI_API_KEY is not set"
            raise ConfigError(msg)
        self._client = genai.Client(api_key=s.gemini_api_key)
        self._model = model_name or s.genai_model_name
        self.max_retries = max_retries

    def complete(self, system: str, user: str, *, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        prompt = f"{system}\n\n{user}"
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                )
                text = (getattr(resp, "text", None) or "").strip()
                return text
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(0.5 * (2**attempt))
        raise RetriableAPIError(str(last)) from last
