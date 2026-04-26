from __future__ import annotations

import logging
import re
import time
from typing import Any

import numpy as np
from google import genai
from google.genai import types

from optirag.config.settings import get_settings
from optirag.domain.errors import ConfigError, RetriableAPIError

logger = logging.getLogger(__name__)


_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)s\s*$")


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
        max_rate_limit_retries: int = 60,
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
        self.max_rate_limit_retries = max_rate_limit_retries
        self._client = genai.Client(api_key=self._api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._embed_batch(texts)

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            msg = "Cannot embed an empty query"
            raise ValueError(msg)
        return self._embed_batch([text])[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        texts = [text for text in texts if text.strip()]
        if not texts:
            return []
        last: Exception | None = None
        attempt = 0
        rate_limit_waits = 0
        while attempt < self.max_retries:
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
                retry_delay = _extract_retry_delay_seconds(e)
                if retry_delay is not None and rate_limit_waits < self.max_rate_limit_retries:
                    rate_limit_waits += 1
                    wait_s = max(1.0, retry_delay + 1.0)
                    logger.warning(
                        "Gemini embed rate-limited; waiting %.1fs before retry (%d/%d)",
                        wait_s,
                        rate_limit_waits,
                        self.max_rate_limit_retries,
                    )
                    time.sleep(wait_s)
                    continue
                time.sleep(0.5 * (2**attempt))
                attempt += 1
        raise RetriableAPIError(str(last)) from last


def _extract_retry_delay_seconds(exc: Exception) -> float | None:
    """Best-effort extraction of provider retry delay from 429 payloads."""
    # google.genai errors often include a response_json with RetryInfo details.
    response_json = getattr(exc, "response_json", None)
    delay = _parse_retry_delay_from_response_json(response_json)
    if delay is not None:
        return delay

    message = str(exc)
    # Common message pattern: "Please retry in 48.0031029s."
    msg_match = re.search(r"Please retry in ([0-9]+(?:\.[0-9]+)?)s", message)
    if msg_match:
        return float(msg_match.group(1))
    # Serialized details pattern: "retryDelay': '48s'" / "retryDelay\": \"48s\""
    detail_match = re.search(r"retryDelay['\"]?\s*:\s*['\"]([0-9]+(?:\.[0-9]+)?s)['\"]", message)
    if detail_match:
        return _parse_seconds_literal(detail_match.group(1))
    # Fallback: treat explicit 429 as rate-limit with a conservative wait.
    if "429" in message and "RESOURCE_EXHAUSTED" in message:
        return 30.0
    return None


def _parse_retry_delay_from_response_json(data: Any) -> float | None:
    if not isinstance(data, dict):
        return None
    err = data.get("error")
    if not isinstance(err, dict):
        return None
    details = err.get("details")
    if not isinstance(details, list):
        return None
    for item in details:
        if not isinstance(item, dict):
            continue
        raw = item.get("retryDelay")
        sec = _parse_seconds_literal(raw)
        if sec is not None:
            return sec
    return None


def _parse_seconds_literal(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    m = _DURATION_RE.match(value)
    if not m:
        return None
    return float(m.group(1))
