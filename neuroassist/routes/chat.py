"""
routes/chat.py

Plain-text Ollama interview loop for NeuroAssist.
"""

from __future__ import annotations

import json
from typing import Any

from flask import Blueprint, current_app, jsonify, request

from config import Config
from db.connection import execute_query
from services.completeness import (
    compute_completeness_score,
    get_field_section_map,
    get_fields_populated,
    update_completeness_in_db,
)
from services.context_builder import build_input_context
from services.llm import (
    LLMResponseError,
    LLMServiceError,
    call_final_report_model,
    call_interview_model,
)
from services.rag import vector_search

chat_bp = Blueprint("chat", __name__)


def _latest_assistant_payload(session_id: str) -> dict[str, Any] | None:
    row = execute_query(
        """
        SELECT content, content_json
        FROM messages
        WHERE session_id = %s AND role = 'assistant'
        ORDER BY created_at DESC, id DESC
        LIMIT 1
        """,
        (session_id,),
        fetch="one",
    )
    if not row:
        return None

    payload = row.get("content_json")
    if isinstance(payload, dict):
        return dict(payload)

    content = str(row.get("content") or "").strip()
    if content:
        return {"mode": "interview", "assistant_message": content}
    return None


def _persist_context_updates(
    *,
    patient_id: str,
    session_id: str,
    context_updates: dict[str, dict[str, Any]],
    covered_fields: list[str],
) -> dict[str, bool]:
    field_section_map = get_field_section_map()
    row = execute_query(
        """
        SELECT symptoms, clinical, genomics, vasari, pathology,
               labs, treatment_history, fields_populated
        FROM patient_context
        WHERE patient_id = %s AND session_id = %s
        """,
        (patient_id, session_id),
        fetch="one",
    ) or {}

    merged = {
        "symptoms": dict(row.get("symptoms") or {}),
        "clinical": dict(row.get("clinical") or {}),
        "genomics": dict(row.get("genomics") or {}),
        "vasari": dict(row.get("vasari") or {}),
        "pathology": dict(row.get("pathology") or {}),
        "labs": dict(row.get("labs") or {}),
        "treatment_history": dict(row.get("treatment_history") or {}),
        "fields_populated": dict(row.get("fields_populated") or {}),
    }

    for section, updates in (context_updates or {}).items():
        if section in merged and isinstance(updates, dict):
            merged[section].update(updates)

    for field in covered_fields or []:
        if field in field_section_map:
            merged["fields_populated"][field] = True

    for section, updates in (context_updates or {}).items():
        if not isinstance(updates, dict):
            continue
        for field in updates:
            if field in field_section_map:
                merged["fields_populated"][field] = True

    execute_query(
        """
        UPDATE patient_context
        SET symptoms = %s::jsonb,
            clinical = %s::jsonb,
            genomics = %s::jsonb,
            vasari = %s::jsonb,
            pathology = %s::jsonb,
            labs = %s::jsonb,
            treatment_history = %s::jsonb,
            fields_populated = %s::jsonb,
            updated_at = NOW()
        WHERE patient_id = %s AND session_id = %s
        """,
        (
            json.dumps(merged["symptoms"]),
            json.dumps(merged["clinical"]),
            json.dumps(merged["genomics"]),
            json.dumps(merged["vasari"]),
            json.dumps(merged["pathology"]),
            json.dumps(merged["labs"]),
            json.dumps(merged["treatment_history"]),
            json.dumps(merged["fields_populated"]),
            patient_id,
            session_id,
        ),
    )

    return {
        str(field): bool(value)
        for field, value in merged["fields_populated"].items()
    }


def _build_rag_query(*, user_message: str, missing_fields: list[str]) -> str:
    pieces: list[str] = []
    if user_message:
        pieces.append(user_message)
    if missing_fields:
        pieces.append("missing clinical topics: " + ", ".join(missing_fields[:5]))
    pieces.append("glioblastoma neuro-oncology imaging genomics pathology interview")
    return " | ".join(piece for piece in pieces if piece)


def _save_diagnostic_report(
    *,
    patient_id: str,
    session_id: str,
    report: dict[str, Any],
    completeness_score: float,
) -> None:
    execute_query(
        """
        INSERT INTO diagnostic_reports (
            patient_id, session_id, who_grade_predicted, diagnosis_label,
            confidence_score, data_completeness, survival_category, survival_score,
            estimated_median_months, factors_favorable, factors_unfavorable,
            treatment_flags, full_report
        )
        VALUES (
            %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s::jsonb, %s::jsonb,
            %s::jsonb, %s::jsonb
        )
        """,
        (
            patient_id,
            session_id,
            report.get("who_grade_predicted"),
            report.get("diagnosis_label"),
            float(report.get("confidence_score") or 0.0),
            float(completeness_score),
            report.get("survival_category"),
            int(report.get("survival_score") or 0),
            report.get("estimated_median_months"),
            json.dumps(report.get("factors_favorable") or []),
            json.dumps(report.get("factors_unfavorable") or []),
            json.dumps(report.get("treatment_flags") or []),
            json.dumps(report),
        ),
    )


@chat_bp.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True) or {}

    session_id = str(data.get("session_id") or "").strip()
    user_message = str(data.get("user_message") or "").strip()

    if not session_id:
        return jsonify({"error": "session_id is required"}), 400

    sess = execute_query(
        "SELECT patient_id FROM sessions WHERE id = %s AND status = 'active'",
        (session_id,),
        fetch="one",
    )
    if not sess:
        return jsonify({"error": "Active session not found"}), 404

    patient_id = str(sess["patient_id"])

    if not user_message:
        latest = _latest_assistant_payload(session_id)
        if latest:
            return jsonify(latest), 200

    if user_message:
        execute_query(
            "INSERT INTO messages (session_id, role, content) VALUES (%s, 'user', %s)",
            (session_id, user_message),
        )

    fields_populated = get_fields_populated(patient_id, session_id)
    _, missing_fields = compute_completeness_score(fields_populated)
    rag_chunks = vector_search(
        _build_rag_query(user_message=user_message, missing_fields=missing_fields),
        top_k=max(2, int(Config.RAG_TOP_K)),
    )

    input_ctx = build_input_context(
        patient_id=patient_id,
        session_id=session_id,
        rag_chunks=rag_chunks,
        missing_fields=missing_fields,
        current_user_message=user_message,
    )
    force_report = input_ctx["question_turn_count"] >= int(Config.LLM_MAX_INTERVIEW_TURNS)

    try:
        interview_response = call_interview_model(
            input_context=input_ctx["input_context_str"],
            conversation_messages=input_ctx["conversation_messages"],
            max_question_turns=int(Config.LLM_MAX_INTERVIEW_TURNS),
        )
    except LLMResponseError as exc:
        current_app.logger.exception("Interview response validation failed")
        return jsonify({"error": "llm_response_invalid", "detail": str(exc)[:500]}), 502
    except LLMServiceError as exc:
        current_app.logger.exception("Ollama service error")
        detail = str(exc)
        if "connect" in detail.lower() or "connection" in detail.lower():
            return (
                jsonify(
                    {
                        "error": "llm_unavailable",
                        "message": f"Ollama not reachable at {current_app.config.get('OLLAMA_BASE_URL', 'http://localhost:11434')}",
                    }
                ),
                503,
            )
        return jsonify({"error": "llm_service_error", "detail": detail[:500]}), 502
    except Exception as exc:
        current_app.logger.exception("Unexpected interview error")
        return jsonify({"error": "internal_error", "detail": str(exc)[:200]}), 500

    merged_fields = _persist_context_updates(
        patient_id=patient_id,
        session_id=session_id,
        context_updates=interview_response.get("context_updates") or {},
        covered_fields=interview_response.get("covered_fields") or [],
    )
    completeness_score, missing_fields = update_completeness_in_db(
        patient_id,
        session_id,
        merged_fields,
    )

    if interview_response.get("ready_for_report") or force_report:
        final_rag_chunks = vector_search(
            _build_rag_query(user_message=user_message, missing_fields=missing_fields),
            top_k=max(2, int(Config.RAG_TOP_K)),
        )
        final_ctx = build_input_context(
            patient_id=patient_id,
            session_id=session_id,
            rag_chunks=final_rag_chunks,
            missing_fields=missing_fields,
            current_user_message=user_message,
        )

        try:
            report_response = call_final_report_model(
                input_context=final_ctx["input_context_str"],
                conversation_messages=final_ctx["conversation_messages"],
            )
        except LLMResponseError as exc:
            current_app.logger.exception("Final report validation failed")
            return jsonify({"error": "llm_response_invalid", "detail": str(exc)[:500]}), 502
        except LLMServiceError as exc:
            current_app.logger.exception("Final report Ollama service error")
            return jsonify({"error": "llm_service_error", "detail": str(exc)[:500]}), 502
        except Exception as exc:
            current_app.logger.exception("Unexpected final report error")
            return jsonify({"error": "internal_error", "detail": str(exc)[:200]}), 500

        execute_query(
            """
            INSERT INTO messages (session_id, role, content, content_json)
            VALUES (%s, 'assistant', %s, %s::jsonb)
            """,
            (session_id, report_response.get("report_text", ""), json.dumps(report_response)),
        )
        _save_diagnostic_report(
            patient_id=patient_id,
            session_id=session_id,
            report=report_response,
            completeness_score=completeness_score,
        )
        report_response["_meta"] = {
            "completeness_score": completeness_score,
            "fields_missing": missing_fields[:5],
            "max_turns_reached": force_report,
        }
        return jsonify(report_response), 200

    execute_query(
        """
        INSERT INTO messages (session_id, role, content, content_json)
        VALUES (%s, 'assistant', %s, %s::jsonb)
        """,
        (session_id, interview_response.get("assistant_message", ""), json.dumps(interview_response)),
    )
    interview_response["_meta"] = {
        "completeness_score": completeness_score,
        "fields_missing": missing_fields[:5],
        "question_turn_count": input_ctx["question_turn_count"] + 1,
    }
    return jsonify(interview_response), 200
