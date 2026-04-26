from __future__ import annotations

from optirag.domain.types import TextChunk
from optirag.optimization.trial_params import Stage1TrialParams
from optirag.preprocessing.text import clean_text


def chunk_passage(
    beir_corpus_id: str,
    raw_text: str,
    p: Stage1TrialParams,
) -> list[TextChunk]:
    text = clean_text(raw_text, p.cleaning_mode)
    if not text.strip():
        return []
    strategy = p.chunk_strategy
    if strategy == "identity_one_vec_per_line":
        if len(text) < p.min_chunk_chars and text:
            return []
        return [TextChunk(beir_corpus_id=beir_corpus_id, text=text, chunk_index=0)]

    size = max(p.chunk_size, 1)
    overlap = min(p.chunk_overlap, size - 1) if size > 1 else 0
    if strategy == "fixed_window":
        return _char_windows(text, beir_corpus_id, size, overlap, fixed_step=True, min_chars=p.min_chunk_chars)
    if strategy == "sliding_window":
        return _char_windows(text, beir_corpus_id, size, overlap, fixed_step=False, min_chars=p.min_chunk_chars)
    if strategy == "recursive":
        return _recursive_chunks(text, beir_corpus_id, size, overlap, p.min_chunk_chars)
    msg = f"Unknown chunk strategy: {strategy}"
    raise ValueError(msg)


def _char_windows(
    text: str,
    beir_corpus_id: str,
    size: int,
    overlap: int,
    *,
    fixed_step: bool,
    min_chars: int,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    step = size - overlap if not fixed_step else size - overlap
    step = max(step, 1)
    i = 0
    idx = 0
    while i < len(text):
        piece = text[i : i + size]
        if len(piece) >= min_chars:
            chunks.append(TextChunk(beir_corpus_id=beir_corpus_id, text=piece, chunk_index=idx))
            idx += 1
        i += step
    if not chunks and text:
        chunks.append(TextChunk(beir_corpus_id=beir_corpus_id, text=text[:size], chunk_index=0))
    return chunks


def _recursive_chunks(
    text: str,
    beir_corpus_id: str,
    size: int,
    overlap: int,
    min_chars: int,
) -> list[TextChunk]:
    """Split on blank lines, then sub-chunk long blocks with fixed windows; merge into one stream."""
    out: list[TextChunk] = []
    for block in text.split("\n\n"):
        b = block.strip()
        if not b:
            continue
        if len(b) <= size:
            if len(b) >= min_chars or not min_chars:
                out.append(TextChunk(beir_corpus_id=beir_corpus_id, text=b, chunk_index=len(out)))
        else:
            out.extend(_char_windows(b, beir_corpus_id, size, overlap, fixed_step=True, min_chars=min_chars))
    if not out and text:
        return [TextChunk(beir_corpus_id=beir_corpus_id, text=text[:size], chunk_index=0)]
    return out


def chunk_corpus(
    corpus: dict[str, str],
    p: Stage1TrialParams,
) -> list[TextChunk]:
    out: list[TextChunk] = []
    for doc_id, raw in corpus.items():
        out.extend(chunk_passage(doc_id, raw, p))
    return out
