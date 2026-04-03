"""
services/tool_runtime.py

Runtime-safe tool registry and execution helpers for chat orchestration.
"""

from __future__ import annotations

import json
from typing import Any

from db.connection import execute_query
from services.rag import vector_search

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "get_patient_snapshot",
        "description": "Fetch the latest structured patient context JSON for the active session.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_imaging_metrics",
        "description": "Fetch latest imaging metrics (volumes, diameter, lobe flags).",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "search_knowledge_base",
        "description": "Semantic search in knowledge chunks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_conversation_chunks",
        "description": "Return conversation split into compact chunks for final reasoning.",
        "input_schema": {
            "type": "object",
            "properties": {
                "chunk_size": {"type": "integer"},
                "max_chunks": {"type": "integer"},
            },
        },
    },
]


def list_tool_specs() -> list[dict[str, Any]]:
    return TOOL_SPECS


def build_conversation_chunks(
    session_id: str,
    chunk_size: int = 12,
    max_chunks: int = 10,
) -> list[dict[str, Any]]:
    chunk_size = max(4, min(30, int(chunk_size)))
    max_chunks = max(1, min(20, int(max_chunks)))

    rows = execute_query(
        """
        SELECT role, content, created_at
        FROM messages
        WHERE session_id = %s
        ORDER BY created_at ASC
        """,
        (session_id,),
        fetch="all",
    ) or []

    lines = []
    for row in rows:
        role = str(row.get("role") or "assistant")
        content = str(row.get("content") or "").strip()
        if not content:
            continue
        if len(content) > 1200:
            content = content[:1200] + "…"
        lines.append({"role": role, "content": content})

    chunks: list[dict[str, Any]] = []
    idx = 1
    for i in range(0, len(lines), chunk_size):
        if len(chunks) >= max_chunks:
            break
        part = lines[i : i + chunk_size]
        chunk_text = "\n".join([f"{item['role']}: {item['content']}" for item in part])
        chunks.append(
            {
                "chunk_id": f"conv_chunk_{idx:03d}",
                "message_count": len(part),
                "text": chunk_text,
            }
        )
        idx += 1

    return chunks


def execute_tool(
    tool_name: str,
    tool_input: dict[str, Any] | None,
    *,
    session_id: str,
    patient_id: str,
) -> dict[str, Any]:
    name = (tool_name or "").strip()
    args = dict(tool_input or {})

    if name == "get_patient_snapshot":
        row = execute_query(
            """
            SELECT symptoms, clinical, genomics, vasari, pathology,
                   labs, treatment_history, fields_populated, completeness_score
            FROM patient_context
            WHERE patient_id = %s AND session_id = %s
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (patient_id, session_id),
            fetch="one",
        ) or {}
        return {
            "tool_name": name,
            "ok": True,
            "data": {
                "symptoms": dict(row.get("symptoms") or {}),
                "clinical": dict(row.get("clinical") or {}),
                "genomics": dict(row.get("genomics") or {}),
                "vasari": dict(row.get("vasari") or {}),
                "pathology": dict(row.get("pathology") or {}),
                "labs": dict(row.get("labs") or {}),
                "treatment_history": dict(row.get("treatment_history") or {}),
                "fields_populated": dict(row.get("fields_populated") or {}),
                "completeness_score": float(row.get("completeness_score") or 0.0),
            },
        }

    if name == "get_imaging_metrics":
        row = execute_query(
            """
            SELECT wt_volume_cm3, ncr_volume_cm3, ed_volume_cm3, et_volume_cm3,
                   diameter_mm, necrosis_ratio, enhancement_ratio, edema_ratio,
                   midline_shift_mm, lobe_frontal, lobe_temporal,
                   lobe_parietal, lobe_occipital, lobe_other
            FROM imaging_reports
            WHERE session_id = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (session_id,),
            fetch="one",
        ) or {}
        return {"tool_name": name, "ok": True, "data": dict(row)}

    if name == "search_knowledge_base":
        query = str(args.get("query") or "").strip()
        top_k = int(args.get("top_k") or 5)
        if not query:
            return {"tool_name": name, "ok": False, "error": "query is required"}
        chunks = vector_search(query, top_k=max(1, min(10, top_k)))
        return {"tool_name": name, "ok": True, "query": query, "chunks": chunks}

    if name == "get_conversation_chunks":
        chunk_size = int(args.get("chunk_size") or 12)
        max_chunks = int(args.get("max_chunks") or 10)
        chunks = build_conversation_chunks(
            session_id=session_id,
            chunk_size=chunk_size,
            max_chunks=max_chunks,
        )
        return {"tool_name": name, "ok": True, "chunks": chunks}

    return {"tool_name": name, "ok": False, "error": f"Unknown tool: {name}"}


def stringify_tool_result(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False)
