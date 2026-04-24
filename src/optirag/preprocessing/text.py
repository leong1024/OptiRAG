from __future__ import annotations

import re
import unicodedata
from typing import Literal

CleaningMode = Literal["none", "light_normalize"]


def clean_text(text: str, mode: CleaningMode) -> str:
    if mode == "none":
        return text
    if mode == "light_normalize":
        t = unicodedata.normalize("NFKC", text)
        t = re.sub(r"\s+", " ", t).strip()
        return t
    return text
