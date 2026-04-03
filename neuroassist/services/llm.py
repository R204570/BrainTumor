"""
services/llm.py

Local Ollama client for the NeuroAssist interview loop.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from config import Config
from services.completeness import get_allowed_fields_by_section

logger = logging.getLogger(__name__)


class LLMResponseError(RuntimeError):
    """Raised when LLM output cannot be parsed as required JSON."""


class LLMServiceError(RuntimeError):
    """Raised for Ollama transport and service failures."""


def _allowed_fields_block() -> str:
    sections = get_allowed_fields_by_section()
    lines: list[str] = []
    for section, fields in sections.items():
        lines.append(f"- {section}: {', '.join(fields)}")
    return "\n".join(lines)


def build_interview_system_prompt(max_question_turns: int) -> str:
    return f"""
You are NeuroAssist, a neuro-oncology clinical interview assistant.

You already have structured imaging findings from a glioblastoma pipeline.
Your job is to interview the clinician in plain text, gather the missing
clinical context that imaging cannot tell you, and decide when enough
information has been collected for a final report.

Interview rules:
- Ask plain-text questions only. No multiple-choice formatting, no bullets of buttons, no tool mentions.
- One turn may include 1 to 3 tightly related sub-questions when that improves efficiency.
- Do not repeat or re-ask anything that is already answered in the conversation history or Q/A summary.
- Keep an internal checklist silently. Do not expose the checklist to the user.
- Prioritize high-yield missing items first.
- If the latest clinician reply answers a tracked field, capture it in `covered_fields` and `context_updates`.
- If a value is unknown, unavailable, or not stated clearly, do not invent it and do not mark it as covered.
- Stop asking questions once you have enough information OR when the interview has reached {max_question_turns} assistant question turns.

Return valid JSON only.

If you need another interview turn, return exactly this shape:
{{
  "mode": "interview",
  "assistant_message": "Plain-text question for the clinician.",
  "covered_fields": ["field_name_from_allowed_list"],
  "context_updates": {{
    "section_name": {{
      "field_name": "value derived from the latest clinician reply only"
    }}
  }},
  "next_topics": ["short topic label"],
  "ready_for_report": false
}}

If the interview is complete, return exactly this shape:
{{
  "mode": "ready_for_report",
  "assistant_message": "",
  "covered_fields": ["field_name_from_allowed_list"],
  "context_updates": {{
    "section_name": {{
      "field_name": "value derived from the latest clinician reply only"
    }}
  }},
  "next_topics": [],
  "ready_for_report": true,
  "report_reason": "Short internal reason why a final report should now be generated."
}}

Allowed `context_updates` sections:
- symptoms
- clinical
- genomics
- vasari
- pathology
- labs
- treatment_history

Allowed tracked fields:
{_allowed_fields_block()}

Critical behavior:
- `assistant_message` must be plain conversational text.
- Never output markdown code fences.
- Never output prose outside the JSON object.
- `covered_fields` must only contain fields learned from the latest clinician message.
- `context_updates` must only include values supported by the latest clinician message.
- `ready_for_report` must be true only when you are confident the interview can stop.
""".strip()


def build_report_system_prompt() -> str:
    return """
You are NeuroAssist generating the final neuro-oncology summary report.

Use both the imaging pipeline context and the full clinician conversation.
Synthesize them into a single final report. Do not ask more questions.

Return valid JSON only in exactly this shape:
{
  "mode": "final_report",
  "report_text": "Full plain-text diagnostic summary report.",
  "who_grade_predicted": "Grade 2 | Grade 3 | Grade 4 | Indeterminate",
  "diagnosis_label": "Primary diagnostic impression",
  "confidence_score": 0.0,
  "survival_category": "short category",
  "survival_score": 0,
  "estimated_median_months": "text estimate",
  "factors_favorable": ["factor"],
  "factors_unfavorable": ["factor"],
  "treatment_flags": [
    {"severity": "info | warning | critical", "text": "flag text"}
  ],
  "recommendation_summary": "Short recommendation summary.",
  "disclaimer": "Clinical decision support disclaimer."
}

Rules:
- `report_text` must explicitly integrate BOTH imaging findings and the conversational clinical history.
- Keep the report clinically readable and specific.
- If evidence is limited, say so in the report and lower confidence.
- Return JSON only with no markdown fences and no extra prose.
""".strip()


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


def _normalize_context_updates(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        return {}

    normalized: dict[str, dict[str, Any]] = {}
    allowed_sections = set(get_allowed_fields_by_section())
    allowed_fields = {
        field
        for fields in get_allowed_fields_by_section().values()
        for field in fields
    }

    for section, updates in value.items():
        if section not in allowed_sections or not isinstance(updates, dict):
            continue
        kept = {
            str(field): field_value
            for field, field_value in updates.items()
            if str(field) in allowed_fields
        }
        if kept:
            normalized[section] = kept
    return normalized


def _normalize_covered_fields(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    allowed_fields = {
        field
        for fields in get_allowed_fields_by_section().values()
        for field in fields
    }
    out: list[str] = []
    for item in value:
        field = str(item).strip()
        if field and field in allowed_fields and field not in out:
            out.append(field)
    return out


def _validate_interview_response(data: dict[str, Any]) -> dict[str, Any]:
    mode = str(data.get("mode") or "").strip().lower()
    if mode not in {"interview", "ready_for_report"}:
        raise LLMResponseError(f"Unexpected interview mode: {mode or 'missing'}")

    normalized = {
        "mode": mode,
        "assistant_message": str(data.get("assistant_message") or "").strip(),
        "covered_fields": _normalize_covered_fields(data.get("covered_fields")),
        "context_updates": _normalize_context_updates(data.get("context_updates")),
        "next_topics": [
            str(item).strip()
            for item in (data.get("next_topics") or [])
            if str(item).strip()
        ]
        if isinstance(data.get("next_topics"), list)
        else [],
        "ready_for_report": bool(data.get("ready_for_report")),
        "report_reason": str(data.get("report_reason") or "").strip(),
    }

    if mode == "interview" and not normalized["assistant_message"]:
        raise LLMResponseError("Interview response must include assistant_message")

    if mode == "ready_for_report":
        normalized["ready_for_report"] = True

    return normalized


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

    return {
        "mode": "final_report",
        "report_text": report_text,
        "who_grade_predicted": normalized_grade,
        "diagnosis_label": diagnosis_label,
        "confidence_score": float(data.get("confidence_score") or 0.0),
        "survival_category": str(data.get("survival_category") or "").strip(),
        "survival_score": int(data.get("survival_score") or 0),
        "estimated_median_months": str(data.get("estimated_median_months") or "").strip(),
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


def _ollama_chat(messages: list[dict[str, str]]) -> str:
    url = Config.OLLAMA_BASE_URL.rstrip("/") + "/api/chat"
    payload = {
        "model": Config.OLLAMA_MODEL,
        "stream": False,
        "format": "json",
        "messages": messages,
        "options": {
            "temperature": Config.LLM_TEMPERATURE,
            "top_p": Config.LLM_TOP_P,
            "num_predict": Config.LLM_MAX_OUTPUT_TOKENS,
        },
    }

    logger.info(
        "[llm.py] Calling Ollama model=%s base_url=%s message_count=%s",
        Config.OLLAMA_MODEL,
        Config.OLLAMA_BASE_URL,
        len(messages),
    )

    timeout = httpx.Timeout(float(Config.LLM_TIMEOUT_SECONDS))
    try:
        response = httpx.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise LLMServiceError(str(exc)) from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise LLMServiceError("Ollama returned non-JSON HTTP payload") from exc

    content = str((((data or {}).get("message") or {}).get("content")) or "").strip()
    if not content:
        raise LLMServiceError("Ollama returned an empty message body")
    return content


def call_interview_model(
    *,
    input_context: str,
    conversation_messages: list[dict[str, str]],
    max_question_turns: int,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": build_interview_system_prompt(max_question_turns)},
        {"role": "system", "content": input_context},
        *conversation_messages,
    ]
    raw_content = _ollama_chat(messages)
    parsed = _extract_json_object(raw_content)
    if not parsed:
        raise LLMResponseError("Could not parse interview response as JSON")
    return _validate_interview_response(parsed)


def call_final_report_model(
    *,
    input_context: str,
    conversation_messages: list[dict[str, str]],
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": build_report_system_prompt()},
        {"role": "system", "content": input_context},
        *conversation_messages,
    ]
    raw_content = _ollama_chat(messages)
    parsed = _extract_json_object(raw_content)
    if not parsed:
        raise LLMResponseError("Could not parse final report response as JSON")
    return _validate_report_response(parsed)
