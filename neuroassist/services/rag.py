"""
services/rag.py

RAG embedding and retrieval helpers for NeuroAssist.
"""

from __future__ import annotations

import json
from typing import Any

from config import Config
from db.connection import execute_query
from services.embeddings import encode_texts


def _vector_literal(values: list[float]) -> str:
    """Convert list[float] to pgvector literal format."""
    return "[" + ",".join(f"{v:.8f}" for v in values) + "]"


def embed_query(query: str) -> list[float]:
    """Embed a user query into a 384-d vector."""
    q = (query or "").strip()
    if not q:
        return []

    return encode_texts([q])[0]


def vector_search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """
    Retrieve top-k RAG chunks from knowledge_chunks.

    Uses pgvector cosine distance when available, falls back to lexical search.
    """
    q = (query or "").strip()
    if not q:
        return []

    top_k = max(1, int(top_k))
    embedding = embed_query(q)

    if embedding:
        vec = _vector_literal(embedding)
        try:
            rows = execute_query(
                """
                SELECT
                    id,
                    source,
                    chunk_text,
                    metadata,
                    1 - (embedding <=> %s::vector) AS score
                FROM knowledge_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (vec, vec, top_k),
                fetch="all",
            )
            if rows:
                return [dict(r) for r in rows]
        except Exception:
            pass

    rows = execute_query(
        """
        SELECT id, source, chunk_text, metadata,
               0.0::float AS score
        FROM knowledge_chunks
        WHERE chunk_text ILIKE %s
        ORDER BY created_at DESC
        LIMIT %s
        """,
        (f"%{q}%", top_k),
        fetch="all",
    )
    return [dict(r) for r in rows] if rows else []


def format_rag_chunks(chunks: list[dict[str, Any]]) -> str:
    """Render retrieved chunks as a compact text block for prompt context."""
    if not chunks:
        return ""

    per_chunk_limit = max(160, int(Config.RAG_CHUNK_CHAR_LIMIT))
    lines: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        source = chunk.get("source") or "unknown"
        text = (chunk.get("chunk_text") or "").strip()
        if not text:
            continue
        if len(text) > per_chunk_limit:
            text = text[: per_chunk_limit - 3].rstrip() + "..."

        metadata = chunk.get("metadata")
        metadata_str = ""
        if isinstance(metadata, dict) and metadata:
            compact_md = {k: metadata[k] for k in list(metadata)[:2]}
            metadata_str = f" metadata={json.dumps(compact_md, ensure_ascii=True)}"

        lines.append(f"[{idx}] source={source}{metadata_str}\n{text}")

    return "\n\n".join(lines)
