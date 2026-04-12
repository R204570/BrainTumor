"""
services/llm.py

Groq-backed LLM client for NeuroAssist final report generation.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import httpx

from config import Config
from rag_prompts import build_report_prompt, build_system_prompt

logger = logging.getLogger(__name__)

REPORT_SHORT_TEXT_LIMIT = 200


class LLMResponseError(RuntimeError):
    """Raised when LLM output cannot be parsed as required JSON."""


class LLMServiceError(RuntimeError):
    """Raised for LLM transport and service failures."""


class LLMRateLimitError(LLMServiceError):
    """Raised when the upstream provider rejects a request due to rate limits."""


def build_report_system_prompt() -> str:
    return build_system_prompt()


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return {}


def _clip_text(value: Any, limit: int = REPORT_SHORT_TEXT_LIMIT) -> str:
    return str(value or "").strip()[:limit]


def _validate_report_response(data: dict[str, Any]) -> dict[str, Any]:
    mode = str(data.get("mode") or "").strip().lower()
    if mode != "final_report":
        raise LLMResponseError(f"Unexpected report mode: {mode or 'missing'}")

    report_text = str(data.get("report_text") or "").strip()
    diagnosis_label = str(data.get("diagnosis_label") or "").strip()
    if not report_text or not diagnosis_label:
        raise LLMResponseError("Final report requires report_text and diagnosis_label")

    treatment_flags = []
    raw_flags = data.get("treatment_flags") or []
    if isinstance(raw_flags, list):
        for item in raw_flags:
            if not isinstance(item, dict):
                continue
            severity = str(item.get("severity") or "info").strip().lower()
            text = str(item.get("text") or item.get("message") or "").strip()
            if text:
                treatment_flags.append(
                    {
                        "severity": severity if severity in {"info", "warning", "critical"} else "info",
                        "text": text,
                    }
                )

    raw_grade = str(data.get("who_grade_predicted") or "").strip().lower()
    if "grade 4" in raw_grade or raw_grade in {"4", "iv", "grade iv"}:
        normalized_grade = "4"
    elif "grade 3" in raw_grade or raw_grade in {"3", "iii", "grade iii"}:
        normalized_grade = "3"
    elif "grade 2" in raw_grade or raw_grade in {"2", "ii", "grade ii"}:
        normalized_grade = "2"
    elif "grade 1" in raw_grade or raw_grade in {"1", "i", "grade i"}:
        normalized_grade = "1"
    else:
        normalized_grade = "Indeterminate"

    confidence_score = float(data.get("confidence_score") or 0.0)
    confidence_score = max(0.0, min(1.0, confidence_score))

    normalized = {
        "mode": "final_report",
        "report_text": report_text,
        "who_grade_predicted": normalized_grade,
        "diagnosis_label": diagnosis_label,
        "confidence_score": confidence_score,
        "survival_category": _clip_text(data.get("survival_category")),
        "survival_score": int(data.get("survival_score") or 0),
        "estimated_median_months": _clip_text(data.get("estimated_median_months")),
        "factors_favorable": [
            str(item).strip()
            for item in (data.get("factors_favorable") or [])
            if str(item).strip()
        ]
        if isinstance(data.get("factors_favorable"), list)
        else [],
        "factors_unfavorable": [
            str(item).strip()
            for item in (data.get("factors_unfavorable") or [])
            if str(item).strip()
        ]
        if isinstance(data.get("factors_unfavorable"), list)
        else [],
        "treatment_flags": treatment_flags,
        "recommendation_summary": str(data.get("recommendation_summary") or "").strip(),
        "disclaimer": str(
            data.get("disclaimer")
            or "NeuroAssist is a clinical decision support tool. Final decisions must be made by a qualified clinician."
        ).strip(),
    }
    normalized["structured_report"] = _normalize_structured_report(
        data.get("structured_report"),
        normalized,
    )
    return normalized


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _normalize_string_map(value: Any, fields: list[str]) -> dict[str, str]:
    source = value if isinstance(value, dict) else {}
    return {field: str(source.get(field) or "Not provided").strip() or "Not provided" for field in fields}


def _normalize_differential_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    items: list[dict[str, Any]] = []
    for entry in value[:3]:
        if not isinstance(entry, dict):
            continue
        diagnosis = str(entry.get("diagnosis") or "").strip()
        if not diagnosis:
            continue
        probability = str(entry.get("probability") or "Unspecified").strip() or "Unspecified"
        items.append(
            {
                "diagnosis": diagnosis,
                "probability": probability,
                "supporting": _normalize_string_list(entry.get("supporting")),
                "against": _normalize_string_list(entry.get("against")),
            }
        )
    return items


def _normalize_structured_report(value: Any, base_report: dict[str, Any]) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}

    header = _normalize_string_map(
        source.get("header"),
        ["patient_case_id", "report_generated", "ai_model", "report_version"],
    )
    if header["patient_case_id"] == "Not provided":
        header["patient_case_id"] = "Unknown"

    patient_summary = _normalize_string_map(
        source.get("patient_clinical_summary"),
        ["demographics", "presenting_symptoms", "relevant_history", "current_medications"],
    )

    imaging_findings_source = source.get("imaging_findings") if isinstance(source.get("imaging_findings"), dict) else {}
    imaging_findings = {
        "tumor_volumes": _normalize_string_list(imaging_findings_source.get("tumor_volumes")),
        "morphology": str(imaging_findings_source.get("morphology") or "Not provided").strip() or "Not provided",
        "localization": str(imaging_findings_source.get("localization") or "Not provided").strip() or "Not provided",
        "radiomics_flags": _normalize_string_list(imaging_findings_source.get("radiomics_flags")),
    }

    molecular_profile = _normalize_string_map(
        source.get("molecular_profile_integration"),
        ["idh_status", "mgmt_methylation", "codeletion_1p19q", "other_markers", "molecular_grade"],
    )

    grade_source = source.get("grade_assessment") if isinstance(source.get("grade_assessment"), dict) else {}
    grade_assessment = {
        "likely_who_grade": str(
            grade_source.get("likely_who_grade") or base_report.get("who_grade_predicted") or "Indeterminate"
        ).strip(),
        "rationale": _normalize_string_list(grade_source.get("rationale")),
        "confidence": str(grade_source.get("confidence") or "Not provided").strip() or "Not provided",
        "confidence_explanation": str(
            grade_source.get("confidence_explanation") or "Not provided"
        ).strip()
        or "Not provided",
    }

    survival_source = source.get("survival_estimate") if isinstance(source.get("survival_estimate"), dict) else {}
    survival_estimate = {
        "reference_class": str(survival_source.get("reference_class") or "Not provided").strip() or "Not provided",
        "median_os": str(
            survival_source.get("median_os") or base_report.get("estimated_median_months") or "Not provided"
        ).strip(),
        "two_year_os": str(survival_source.get("two_year_os") or "Not provided").strip() or "Not provided",
        "favorable_factors": _normalize_string_list(survival_source.get("favorable_factors"))
        or list(base_report.get("factors_favorable") or []),
        "adverse_factors": _normalize_string_list(survival_source.get("adverse_factors"))
        or list(base_report.get("factors_unfavorable") or []),
        "literature_basis": _normalize_string_list(survival_source.get("literature_basis")),
    }

    treatment_considerations = _normalize_string_map(
        source.get("treatment_considerations"),
        ["surgical", "radiation", "chemotherapy", "monitoring", "mdt_referral"],
    )

    uncertainty_source = (
        source.get("uncertainty_limitations")
        if isinstance(source.get("uncertainty_limitations"), dict)
        else {}
    )
    uncertainty_limitations = {
        "model_limitations": _normalize_string_list(uncertainty_source.get("model_limitations")),
        "missing_data": _normalize_string_list(uncertainty_source.get("missing_data")),
        "recommended_tests": _normalize_string_list(uncertainty_source.get("recommended_tests")),
    }

    summary_for_mdt = _normalize_string_list(source.get("summary_for_mdt"))
    structured = {
        "header": header,
        "patient_clinical_summary": patient_summary,
        "imaging_findings": imaging_findings,
        "differential_diagnosis": _normalize_differential_list(source.get("differential_diagnosis")),
        "molecular_profile_integration": molecular_profile,
        "grade_assessment": grade_assessment,
        "survival_estimate": survival_estimate,
        "treatment_considerations": treatment_considerations,
        "uncertainty_limitations": uncertainty_limitations,
        "summary_for_mdt": summary_for_mdt,
    }

    if not structured["differential_diagnosis"]:
        structured["differential_diagnosis"] = [
            {
                "diagnosis": base_report.get("diagnosis_label") or "Primary glial neoplasm",
                "probability": "Moderate",
                "supporting": [],
                "against": [],
            }
        ]

    return structured


def _groq_chat(
    messages: list[dict[str, str]],
    *,
    num_predict: int | None = None,
) -> str:
    if not Config.GROQ_API_KEY:
        raise LLMServiceError("GROQ_API_KEY is not configured")

    url = Config.GROQ_BASE_URL.rstrip("/") + "/chat/completions"
    predict_limit = int(num_predict or Config.LLM_MAX_OUTPUT_TOKENS)
    payload = {
        "model": Config.GROQ_MODEL,
        "stream": False,
        "messages": messages,
        "temperature": Config.LLM_TEMPERATURE,
        "top_p": Config.LLM_TOP_P,
        "max_completion_tokens": predict_limit,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {Config.GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    total_chars = sum(len(str(message.get("content") or "")) for message in messages)

    logger.info(
        "[llm.py] Calling provider=%s model=%s base_url=%s message_count=%s total_chars=%s num_predict=%s",
        Config.LLM_PROVIDER,
        Config.GROQ_MODEL,
        Config.GROQ_BASE_URL,
        len(messages),
        total_chars,
        predict_limit,
    )

    timeout = httpx.Timeout(float(Config.LLM_TIMEOUT_SECONDS))
    for attempt in range(2):
        try:
            response = httpx.post(url, json=payload, headers=headers, timeout=timeout)
            response.raise_for_status()
            break
        except httpx.HTTPStatusError as exc:
            response_text = ""
            try:
                response_text = exc.response.text.strip()
            except Exception:
                response_text = ""

            if exc.response.status_code == 429:
                retry_after = exc.response.headers.get("retry-after", "").strip()
                wait_seconds = 2.0
                try:
                    if retry_after:
                        wait_seconds = max(1.0, float(retry_after))
                except ValueError:
                    wait_seconds = 2.0

                detail = f"{exc} | body={response_text[:500]}" if response_text else str(exc)
                if attempt == 0:
                    logger.warning("[llm.py] Groq rate-limited; retrying after %.1fs: %s", wait_seconds, detail)
                    time.sleep(wait_seconds)
                    continue
                raise LLMRateLimitError(detail) from exc

            detail = f"{exc}"
            if response_text:
                detail = f"{detail} | body={response_text[:500]}"
            raise LLMServiceError(detail) from exc
        except httpx.HTTPError as exc:
            raise LLMServiceError(str(exc)) from exc
    else:
        raise LLMServiceError("Groq request failed before receiving a response")

    try:
        data = response.json()
    except ValueError as exc:
        raise LLMServiceError("Groq returned non-JSON HTTP payload") from exc

    choices = data.get("choices") or []
    first_choice = choices[0] if isinstance(choices, list) and choices else {}
    message = first_choice.get("message") if isinstance(first_choice, dict) else {}
    content = str((message or {}).get("content") or "").strip()
    if not content:
        raise LLMServiceError("Groq returned an empty message body")
    return content


def _build_report_messages(
    *,
    input_context: str,
    conversation_messages: list[dict[str, str]],
    patient_id: str,
    compact: bool,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": build_report_system_prompt()},
        {
            "role": "user",
            "content": build_report_prompt(
                input_context,
                conversation_messages,
                patient_id=patient_id,
                compact=compact,
            ),
        },
    ]


def call_final_report_model(
    *,
    input_context: str,
    conversation_messages: list[dict[str, str]],
) -> dict[str, Any]:
    patient_id_match = re.search(r"- Patient ID:\s*(.+)", input_context)
    patient_id = patient_id_match.group(1).strip() if patient_id_match else "Unknown"
    first_error: Exception | None = None

    attempts = [
        {
            "compact": False,
            "num_predict": min(int(Config.LLM_MAX_OUTPUT_TOKENS), 1280),
            "label": "primary",
        },
        {
            "compact": True,
            "num_predict": min(int(Config.LLM_MAX_OUTPUT_TOKENS), 896),
            "label": "fallback",
        },
    ]

    for attempt in attempts:
        try:
            messages = _build_report_messages(
                input_context=input_context,
                conversation_messages=conversation_messages,
                patient_id=patient_id,
                compact=bool(attempt["compact"]),
            )
            raw_content = _groq_chat(messages, num_predict=int(attempt["num_predict"]))
            parsed = _extract_json_object(raw_content)
            if not parsed:
                raise LLMResponseError(
                    f"Could not parse final report response as JSON during {attempt['label']} attempt"
                )
            return _validate_report_response(parsed)
        except (LLMResponseError, LLMServiceError) as exc:
            first_error = first_error or exc
            logger.warning(
                "[llm.py] Final report %s attempt failed: %s",
                attempt["label"],
                exc,
            )

    raise LLMServiceError(str(first_error or "Final report generation failed"))
