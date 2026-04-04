"""
routes/chat.py

Form-first report generation endpoint for NeuroAssist.
"""

from __future__ import annotations

from datetime import date
import json
from typing import Any

from flask import Blueprint, current_app, jsonify, request

from config import Config
from db.connection import execute_query
from services.completeness import update_completeness_in_db
from services.context_builder import build_input_context
from services.llm import LLMRateLimitError, LLMResponseError, LLMServiceError, call_final_report_model
from services.rag import vector_search

chat_bp = Blueprint("chat", __name__)

PATIENT_CONTEXT_SECTIONS = (
    "symptoms",
    "clinical",
    "genomics",
    "vasari",
    "pathology",
    "labs",
    "treatment_history",
)


def _age_from_dob(dob: Any) -> int | None:
    if isinstance(dob, date):
        today = date.today()
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

    if isinstance(dob, str) and dob:
        try:
            parsed = date.fromisoformat(dob)
            today = date.today()
            return today.year - parsed.year - ((today.month, today.day) < (parsed.month, parsed.day))
        except ValueError:
            return None

    return None


def _hydrate_registered_intake_context(patient_id: str, session_id: str, *, dob: Any, sex: str) -> None:
    row = execute_query(
        """
        SELECT clinical, fields_populated
        FROM patient_context
        WHERE patient_id = %s AND session_id = %s
        """,
        (patient_id, session_id),
        fetch="one",
    ) or {}

    clinical = dict(row.get("clinical") or {})
    fields_populated = dict(row.get("fields_populated") or {})
    changed = False

    age = _age_from_dob(dob)
    if age is not None:
        if not clinical.get("age_at_diagnosis"):
            clinical["age_at_diagnosis"] = age
            changed = True
        if not fields_populated.get("age_at_diagnosis"):
            fields_populated["age_at_diagnosis"] = True
            changed = True

    normalized_sex = str(sex or "").strip()
    if normalized_sex:
        if not clinical.get("sex"):
            clinical["sex"] = normalized_sex
            changed = True
        if not fields_populated.get("sex"):
            fields_populated["sex"] = True
            changed = True

    if not changed:
        return

    execute_query(
        """
        UPDATE patient_context
        SET clinical = %s::jsonb,
            fields_populated = %s::jsonb,
            updated_at = NOW()
        WHERE patient_id = %s AND session_id = %s
        """,
        (
            json.dumps(clinical),
            json.dumps(fields_populated),
            patient_id,
            session_id,
        ),
    )


def _persist_context_updates(
    *,
    patient_id: str,
    session_id: str,
    context_updates: dict[str, dict[str, Any]],
) -> dict[str, bool]:
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
            for field in updates:
                merged["fields_populated"][str(field)] = True

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

    return {str(field): bool(value) for field, value in merged["fields_populated"].items()}


def _build_rag_query(*, intake_summary: str, missing_fields: list[str]) -> str:
    pieces: list[str] = []
    if intake_summary:
        pieces.append(intake_summary[:1200])
    if missing_fields:
        pieces.append("missing clinical topics: " + ", ".join(missing_fields[:5]))
    pieces.append("glioblastoma neuro-oncology imaging genomics pathology clinical intake")
    return " | ".join(piece for piece in pieces if piece)


def _value_present(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    return True


def _clean_context_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return value


def _normalize_intake_context_updates(raw: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, dict):
        return {}

    normalized: dict[str, dict[str, Any]] = {}
    for section in PATIENT_CONTEXT_SECTIONS:
        source = raw.get(section)
        if not isinstance(source, dict):
            continue
        cleaned: dict[str, Any] = {}
        for key, value in source.items():
            key_text = str(key).strip()
            if not key_text:
                continue
            cleaned_value = _clean_context_value(value)
            if _value_present(cleaned_value):
                cleaned[key_text] = cleaned_value
        if cleaned:
            normalized[section] = cleaned
    return normalized


def _build_intake_summary(
    *,
    session_id: str,
    context_updates: dict[str, dict[str, Any]],
) -> str:
    lines = [
        "NEUROASSIST CLINICAL INTAKE FORM SUBMISSION",
        f"Session ID: {session_id}",
        "",
        "The following data was submitted via the structured intake form and should be treated as verified clinician-provided context.",
        "",
    ]

    for section in PATIENT_CONTEXT_SECTIONS:
        items = context_updates.get(section) or {}
        if not items:
            continue
        lines.append(section.upper())
        for key, value in items.items():
            rendered = ", ".join(str(item) for item in value) if isinstance(value, list) else str(value)
            lines.append(f"- {key}: {rendered or 'Not provided'}")
        lines.append("")

    return "\n".join(lines).strip()


def _validate_intake_submission(
    *,
    context_updates: dict[str, dict[str, Any]],
    session_row: dict[str, Any],
) -> list[str]:
    clinical = context_updates.get("clinical") or {}
    symptoms = context_updates.get("symptoms") or {}

    errors: list[str] = []
    registered_age = _age_from_dob(session_row.get("date_of_birth"))
    if not _value_present(clinical.get("age_at_diagnosis")) and registered_age is None:
        errors.append("Age is required.")
    if not _value_present(clinical.get("sex")) and not str(session_row.get("sex") or "").strip():
        errors.append("Biological sex is required.")
    if not (
        _value_present(symptoms.get("symptom_duration_days"))
        or _value_present(symptoms.get("presenting_symptoms"))
        or _value_present(symptoms.get("neurological_exam"))
    ):
        errors.append("At least one symptom or neurological presentation field is required.")

    return errors


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


@chat_bp.route("/session/<session_id>/generate-report", methods=["POST"])
def generate_report_from_intake(session_id: str):
    data = request.get_json(force=True) or {}

    sess = execute_query(
        """
        SELECT s.patient_id, p.date_of_birth, p.sex
        FROM sessions s
        JOIN patients p ON p.id = s.patient_id
        WHERE s.id = %s AND s.status = 'active'
        """,
        (session_id,),
        fetch="one",
    )
    if not sess:
        return jsonify({"error": "Active session not found"}), 404

    imaging_exists = execute_query(
        """
        SELECT id
        FROM imaging_reports
        WHERE session_id = %s
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (session_id,),
        fetch="one",
    )
    if not imaging_exists:
        return jsonify({"error": "MRI inference must complete before generating the report."}), 400

    patient_id = str(sess["patient_id"])
    _hydrate_registered_intake_context(
        patient_id,
        session_id,
        dob=sess.get("date_of_birth"),
        sex=str(sess.get("sex") or ""),
    )

    context_updates = _normalize_intake_context_updates(data.get("context_updates"))
    validation_errors = _validate_intake_submission(
        context_updates=context_updates,
        session_row=dict(sess),
    )
    if validation_errors:
        return jsonify({"error": "validation_error", "detail": validation_errors}), 400

    merged_fields = _persist_context_updates(
        patient_id=patient_id,
        session_id=session_id,
        context_updates=context_updates,
    )
    completeness_score, missing_fields = update_completeness_in_db(
        patient_id,
        session_id,
        merged_fields,
    )

    intake_summary = _build_intake_summary(
        session_id=session_id,
        context_updates=context_updates,
    )
    rag_chunks = vector_search(
        _build_rag_query(intake_summary=intake_summary, missing_fields=missing_fields),
        top_k=max(1, min(2, int(Config.RAG_TOP_K))),
    )
    input_ctx = build_input_context(
        patient_id=patient_id,
        session_id=session_id,
        rag_chunks=rag_chunks,
        missing_fields=missing_fields,
        current_user_message="",
        include_message_history=False,
    )

    try:
        report_response = call_final_report_model(
            input_context=input_ctx["input_context_str"],
            conversation_messages=[{"role": "user", "content": intake_summary}],
        )
    except LLMResponseError as exc:
        current_app.logger.exception("Final report validation failed")
        return jsonify({"error": "llm_response_invalid", "detail": str(exc)[:500]}), 502
    except LLMRateLimitError as exc:
        current_app.logger.exception("Final report provider rate limit")
        return jsonify({"error": "llm_rate_limit", "detail": str(exc)[:500]}), 429
    except LLMServiceError as exc:
        current_app.logger.exception("Final report provider service error")
        return jsonify({"error": "llm_service_error", "detail": str(exc)[:500]}), 502
    except Exception as exc:
        current_app.logger.exception("Unexpected final report error")
        return jsonify({"error": "internal_error", "detail": str(exc)[:200]}), 500

    try:
        execute_query(
            """
            INSERT INTO messages (session_id, role, content)
            VALUES (%s, 'user', %s)
            """,
            (session_id, intake_summary),
        )
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
    except Exception as exc:
        current_app.logger.exception("Failed while persisting final report")
        return jsonify({"error": "report_persistence_error", "detail": str(exc)[:500]}), 500

    report_response["_meta"] = {
        "completeness_score": completeness_score,
        "fields_missing": missing_fields[:5],
        "mode": "form_submission",
    }
    return jsonify(report_response), 200
