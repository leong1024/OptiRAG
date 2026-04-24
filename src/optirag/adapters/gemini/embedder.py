from __future__ import annotations

import time
from typing import Any

import numpy as np
from google import genai
from google.genai import types

from optirag.config.settings import get_settings
from optirag.domain.errors import ConfigError, RetriableAPIError


def _l2_normalize(vec: list[float]) -> list[float]:
    a = np.array(vec, dtype=np.float64)
    n = np.linalg.norm(a)
    if n == 0:
        return vec
    return (a / n).tolist()


class GeminiEmbedder:
    def __init__(
        self,
        model_id: str,
        *,
        output_dimensionality: int | None = None,
        l2_normalize: bool = True,
        max_retries: int = 3,
    ) -> None:
        s = get_settings()
        if not s.gemini_api_key:
            msg = "GEMINI_API_KEY is not set"
            raise ConfigError(msg)
        self._api_key = s.gemini_api_key
        self.model_id = model_id
        self.output_dimensionality = output_dimensionality
        self.l2_normalize = l2_normalize
        self.max_retries = max_retries
        self._client = genai.Client(api_key=self._api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._embed_batch(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                cfg: types.EmbedContentConfig | None = None
                if self.output_dimensionality is not None:
                    cfg = types.EmbedContentConfig(output_dimensionality=self.output_dimensionality)
                kwargs: dict[str, Any] = {
                    "model": self.model_id,
                    "contents": texts,
                }
                if cfg is not None:
                    kwargs["config"] = cfg
                result = self._client.models.embed_content(**kwargs)
                embs = result.embeddings
                out: list[list[float]] = []
                for e in embs:
                    vec = list(e.values)  # type: ignore[union-attr]
                    out.append(_l2_normalize(vec) if self.l2_normalize else vec)
                return out
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(0.5 * (2**attempt))
        raise RetriableAPIError(str(last)) from last
