"""
services/embeddings.py

Embedding utilities with graceful fallback when sentence-transformers is unavailable.
"""

from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache
from typing import Iterable

from config import Config

EMBED_DIM = 384


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 1e-12:
        return [0.0] * len(vec)
    return [v / norm for v in vec]


def _fallback_embed(text: str, dim: int = EMBED_DIM) -> list[float]:
    """Deterministic hashed embedding fallback (no external model dependency)."""
    vec = [0.0] * dim
    tokens = re.findall(r"[A-Za-z0-9_]+", (text or "").lower())
    if not tokens:
        tokens = [text or "empty"]

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8", errors="ignore")).digest()
        for i in range(0, len(digest), 2):
            idx = ((digest[i] << 8) + digest[i + 1]) % dim
            sign = 1.0 if (digest[i] % 2 == 0) else -1.0
            vec[idx] += sign

    return _normalize(vec)


@lru_cache(maxsize=1)
def _load_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(Config.EMBED_MODEL)
    except Exception:
        return None


def encode_texts(texts: Iterable[str]) -> list[list[float]]:
    data = [str(t or "") for t in texts]
    if not data:
        return []

    model = _load_sentence_transformer()
    if model is not None:
        try:
            vectors = model.encode(data, normalize_embeddings=True)
            return [[float(x) for x in row] for row in vectors]
        except Exception:
            pass

    return [_fallback_embed(text) for text in data]
