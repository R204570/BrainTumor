"""
services/context_builder.py

Builds the full dynamic context passed to the Ollama interview/report prompts.
"""

from __future__ import annotations

from typing import Any

from config import Config
from data_pipeline import format_pipeline_context
from db.connection import execute_query
from services.rag import format_rag_chunks


def _latest_pipeline_output(session_id: str) -> dict[str, Any]:
    row = execute_query(
        """
        SELECT wt_volume_cm3, ncr_volume_cm3, ed_volume_cm3, et_volume_cm3,
               diameter_mm,
               centroid_x, centroid_y, centroid_z,
               lobe_pct_frontal, lobe_pct_temporal, lobe_pct_parietal, lobe_pct_occipital, lobe_pct_other,
               necrosis_ratio, enhancement_ratio, edema_ratio, midline_shift_mm
        FROM imaging_reports
        WHERE session_id = %s
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (session_id,),
        fetch="one",
    ) or {}

    return {
        "wt_volume_cm3": float(row.get("wt_volume_cm3") or 0.0),
        "ncr_volume_cm3": float(row.get("ncr_volume_cm3") or 0.0),
        "ed_volume_cm3": float(row.get("ed_volume_cm3") or 0.0),
        "et_volume_cm3": float(row.get("et_volume_cm3") or 0.0),
        "diameter_mm": float(row.get("diameter_mm") or 0.0),
        "centroid": (
            int(row.get("centroid_z") or 64),
            int(row.get("centroid_y") or 64),
            int(row.get("centroid_x") or 64),
        ),
        "lobe_involvement": {
            "Frontal": float(row.get("lobe_pct_frontal") or 0.0),
            "Temporal": float(row.get("lobe_pct_temporal") or 0.0),
            "Parietal": float(row.get("lobe_pct_parietal") or 0.0),
            "Occipital": float(row.get("lobe_pct_occipital") or 0.0),
            "Other": float(row.get("lobe_pct_other") or 0.0),
        },
        "derived": {
            "necrosis_ratio": float(row.get("necrosis_ratio") or 0.0),
            "enhancement_ratio": float(row.get("enhancement_ratio") or 0.0),
            "edema_ratio": float(row.get("edema_ratio") or 0.0),
            "midline_shift_mm": float(row.get("midline_shift_mm") or 0.0),
        },
    }


def _load_patient_context(patient_id: str, session_id: str) -> dict[str, Any]:
    row = execute_query(
        """
        SELECT symptoms, clinical, genomics, vasari, pathology,
               labs, treatment_history, fields_populated, completeness_score
        FROM patient_context
        WHERE patient_id = %s AND session_id = %s
        """,
        (patient_id, session_id),
        fetch="one",
    ) or {}

    return {
        "symptoms": dict(row.get("symptoms") or {}),
        "clinical": dict(row.get("clinical") or {}),
        "genomics": dict(row.get("genomics") or {}),
        "vasari": dict(row.get("vasari") or {}),
        "pathology": dict(row.get("pathology") or {}),
        "labs": dict(row.get("labs") or {}),
        "treatment_history": dict(row.get("treatment_history") or {}),
        "fields_populated": dict(row.get("fields_populated") or {}),
        "completeness_score": float(row.get("completeness_score") or 0.0),
    }


def _format_patient_context_block(patient_context: dict[str, Any]) -> str:
    lines: list[str] = []
    for section in [
        "symptoms",
        "clinical",
        "genomics",
        "vasari",
        "pathology",
        "labs",
        "treatment_history",
    ]:
        data = patient_context.get(section) or {}
        if not data:
            continue
        lines.append(f"- {section}: {data}")
    return "\n".join(lines) if lines else "- No structured patient context captured yet"


def _load_message_rows(session_id: str) -> list[dict[str, Any]]:
    return execute_query(
        """
        SELECT role, content, content_json, created_at
        FROM messages
        WHERE session_id = %s
        ORDER BY created_at ASC, id ASC
        LIMIT %s
        """,
        (session_id, max(200, int(Config.CONTEXT_HISTORY_LIMIT) * 4)),
        fetch="all",
    ) or []


def _assistant_text_from_row(row: dict[str, Any]) -> str:
    payload = row.get("content_json")
    if isinstance(payload, dict):
        mode = str(payload.get("mode") or "").strip().lower()
        if mode == "interview":
            return str(payload.get("assistant_message") or row.get("content") or "").strip()
        if mode == "final_report":
            return str(payload.get("report_text") or row.get("content") or "").strip()
    return str(row.get("content") or "").strip()


def _conversation_messages(session_id: str) -> list[dict[str, str]]:
    rows = _load_message_rows(session_id)
    messages: list[dict[str, str]] = []
    for row in rows:
        role = str(row.get("role") or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = _assistant_text_from_row(row) if role == "assistant" else str(row.get("content") or "").strip()
        if not content:
            continue
        messages.append({"role": role, "content": content})
    return messages


def _question_answer_pairs(session_id: str) -> list[dict[str, str]]:
    rows = _load_message_rows(session_id)
    pairs: list[dict[str, str]] = []
    pending_question: str | None = None

    for row in rows:
        role = str(row.get("role") or "").strip().lower()
        if role == "assistant":
            payload = row.get("content_json")
            mode = str((payload or {}).get("mode") or "").strip().lower() if isinstance(payload, dict) else ""
            if mode == "interview":
                text = _assistant_text_from_row(row)
                if text:
                    pending_question = text
        elif role == "user":
            answer = str(row.get("content") or "").strip()
            if not answer:
                continue
            if pending_question:
                pairs.append({"question": pending_question, "answer": answer})
                pending_question = None
            else:
                pairs.append({"question": "(clinician note)", "answer": answer})

    return pairs


def _format_previous_pairs(pairs: list[dict[str, str]]) -> str:
    if not pairs:
        return "None yet."

    lines: list[str] = []
    for idx, pair in enumerate(pairs, start=1):
        lines.append(f"{idx}. Q: {pair['question']}")
        lines.append(f"   A: {pair['answer']}")
    return "\n".join(lines)


def _current_turn_block(current_user_message: str, qa_pairs: list[dict[str, str]]) -> str:
    if not current_user_message:
        return (
            "Current turn:\n"
            "- No new clinician answer in this request.\n"
            "- If the interview has not started yet, ask the first high-yield question."
        )

    current_pair = qa_pairs[-1] if qa_pairs else None
    if current_pair and current_pair.get("answer") == current_user_message:
        question_text = current_pair.get("question") or "(no prior assistant question found)"
    else:
        question_text = "(clinician provided an unsolicited note rather than a direct answer)"

    return (
        "Current turn:\n"
        f"- Present question: {question_text}\n"
        f"- Current answer: {current_user_message}"
    )


def build_input_context(
    *,
    patient_id: str,
    session_id: str,
    rag_chunks: list[dict[str, Any]],
    missing_fields: list[str],
    current_user_message: str = "",
) -> dict[str, Any]:
    patient_context = _load_patient_context(patient_id, session_id)
    pipeline_output = _latest_pipeline_output(session_id)
    qa_pairs = _question_answer_pairs(session_id)
    previous_pairs = qa_pairs[:-1] if current_user_message and qa_pairs else qa_pairs
    conversation_messages = _conversation_messages(session_id)

    question_turn_count = 0
    for row in _load_message_rows(session_id):
        payload = row.get("content_json")
        if isinstance(payload, dict) and str(payload.get("mode") or "").strip().lower() == "interview":
            question_turn_count += 1

    completeness_pct = int(round(float(patient_context.get("completeness_score") or 0.0) * 100))
    rag_text = format_rag_chunks(rag_chunks) or "No additional RAG references retrieved."
    imaging_context = format_pipeline_context(pipeline_output)
    imaging_derived = pipeline_output.get("derived") or {}

    input_context_str = (
        "NEUROASSIST DYNAMIC CONTEXT\n"
        f"- Session ID: {session_id}\n"
        f"- Interview turns already used: {question_turn_count}\n"
        f"- Max interview turns: {int(getattr(Config, 'LLM_MAX_INTERVIEW_TURNS', 15))}\n"
        f"- Structured completeness score: {completeness_pct}%\n"
        f"- Missing tracked fields: {', '.join(missing_fields[:10]) if missing_fields else 'None currently tracked as missing'}\n\n"
        f"{imaging_context}\n"
        f"- Derived imaging notes: necrosis_ratio={imaging_derived.get('necrosis_ratio', 0.0):.3f}, "
        f"enhancement_ratio={imaging_derived.get('enhancement_ratio', 0.0):.3f}, "
        f"edema_ratio={imaging_derived.get('edema_ratio', 0.0):.3f}, "
        f"midline_shift_mm={imaging_derived.get('midline_shift_mm', 0.0):.2f}\n\n"
        f"{_current_turn_block(current_user_message, qa_pairs)}\n\n"
        "Previous asked questions and answers:\n"
        f"{_format_previous_pairs(previous_pairs)}\n\n"
        "Structured patient context already captured:\n"
        f"{_format_patient_context_block(patient_context)}\n\n"
        "Relevant RAG knowledge snippets:\n"
        f"{rag_text}\n\n"
        "Important reminder:\n"
        "- Use the current answer together with all previous Q/A.\n"
        "- Do not re-ask answered topics.\n"
        "- Ask the next best plain-text question unless it is time to stop and prepare the report."
    )

    return {
        "patient_context": patient_context,
        "pipeline_output": pipeline_output,
        "conversation_messages": conversation_messages,
        "question_answer_pairs": qa_pairs,
        "question_turn_count": question_turn_count,
        "input_context_str": input_context_str,
    }
