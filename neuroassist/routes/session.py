"""
routes/session.py

Session endpoints:
- GET  /                          -> dashboard/new session page
- POST /session/new               -> create patient + session + context row
- GET  /session/<session_id>      -> Sprint 3 session view
- GET  /session/<session_id>/state -> live session state JSON for UI
"""

from __future__ import annotations

from datetime import date, datetime
import json
from typing import Any

from flask import Blueprint, jsonify, redirect, render_template, request, url_for

from db.connection import execute_query
from services.completeness import get_all_field_weights

session_bp = Blueprint("session", __name__)


@session_bp.route("/")
def index():
    """Render dashboard/new session page."""
    return render_template("index.html")


@session_bp.route("/records")
def records_view():
    """Searchable patient record directory with session/report shortcuts."""
    query = (request.args.get("q") or "").strip()
    patients = _load_patient_records(query)
    session_count = sum(len(patient["sessions"]) for patient in patients)

    return render_template(
        "records.html",
        query=query,
        patients=patients,
        patient_count=len(patients),
        session_count=session_count,
    )


@session_bp.route("/session/new", methods=["POST"])
def new_session():
    """Create patient + session and initialize patient_context."""
    data = request.get_json(force=True) if request.is_json else request.form.to_dict()

    first_name = (data.get("first_name") or "").strip()
    last_name = (data.get("last_name") or "").strip()
    mrn = (data.get("mrn") or "").strip() or None
    date_of_birth = data.get("date_of_birth") or None
    sex = (data.get("sex") or "").strip() or None

    if not first_name:
        return jsonify({"error": "first_name is required"}), 400

    if mrn:
        patient_row = execute_query(
            """
            INSERT INTO patients (mrn, first_name, last_name, date_of_birth, sex)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (mrn)
            DO UPDATE SET
                first_name = EXCLUDED.first_name,
                last_name = EXCLUDED.last_name,
                date_of_birth = COALESCE(EXCLUDED.date_of_birth, patients.date_of_birth),
                sex = COALESCE(EXCLUDED.sex, patients.sex)
            RETURNING id
            """,
            (mrn, first_name, last_name, date_of_birth, sex),
            fetch="one",
        )
    else:
        patient_row = execute_query(
            """
            INSERT INTO patients (mrn, first_name, last_name, date_of_birth, sex)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (mrn, first_name, last_name, date_of_birth, sex),
            fetch="one",
        )

    patient_id = str(patient_row["id"])

    session_row = execute_query(
        """
        INSERT INTO sessions (patient_id, status)
        VALUES (%s, 'active')
        RETURNING id
        """,
        (patient_id,),
        fetch="one",
    )
    session_id = str(session_row["id"])

    initial_clinical: dict[str, Any] = {}
    initial_fields_populated: dict[str, bool] = {}
    age_at_diagnosis = _age_from_dob(date_of_birth)
    if age_at_diagnosis is not None:
        initial_clinical["age_at_diagnosis"] = age_at_diagnosis
        initial_fields_populated["age_at_diagnosis"] = True
    if sex:
        initial_clinical["sex"] = sex
        initial_fields_populated["sex"] = True

    execute_query(
        """
        INSERT INTO patient_context (patient_id, session_id, clinical, completeness_score, fields_populated)
        VALUES (%s, %s, %s::jsonb, 0.000, %s::jsonb)
        """,
        (
            patient_id,
            session_id,
            json.dumps(initial_clinical),
            json.dumps(initial_fields_populated),
        ),
    )

    if request.is_json:
        return jsonify({"patient_id": patient_id, "session_id": session_id}), 201

    return redirect(url_for("session.session_view", session_id=session_id))


@session_bp.route("/session/<session_id>")
def session_view(session_id: str):
    """Render Sprint 3 main session view."""
    row = execute_query(
        """
        SELECT s.id AS session_id,
               s.status,
               p.first_name,
               p.last_name,
               p.mrn
        FROM sessions s
        JOIN patients p ON p.id = s.patient_id
        WHERE s.id = %s
        """,
        (session_id,),
        fetch="one",
    )

    if not row:
        return "Session not found", 404

    return render_template("session.html", session=dict(row), session_id=session_id)


@session_bp.route("/session/<session_id>/state", methods=["GET"])
def session_state(session_id: str):
    """Return live session state for Sprint 3 frontend."""
    base = execute_query(
        """
        SELECT s.id AS session_id,
               s.status,
               s.created_at,
               p.id AS patient_id,
               p.first_name,
               p.last_name,
               p.mrn,
               p.date_of_birth,
               p.sex,
               pc.completeness_score,
               pc.symptoms,
               pc.clinical,
               pc.genomics,
               pc.vasari,
               pc.pathology,
               pc.labs,
               pc.treatment_history,
               pc.fields_populated
        FROM sessions s
        JOIN patients p ON p.id = s.patient_id
        LEFT JOIN patient_context pc ON pc.session_id = s.id
        WHERE s.id = %s
        """,
        (session_id,),
        fetch="one",
    )

    if not base:
        return jsonify({"error": "Session not found"}), 404

    imaging = execute_query(
        """
        SELECT wt_volume_cm3, necrosis_ratio, edema_ratio, midline_shift_mm,
               ncr_volume_cm3, ed_volume_cm3, et_volume_cm3, diameter_mm,
               lobe_frontal, lobe_temporal, lobe_parietal, lobe_occipital, lobe_other
        FROM imaging_reports
        WHERE session_id = %s
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (session_id,),
        fetch="one",
    ) or {}

    latest_report = execute_query(
        """
        SELECT who_grade_predicted, confidence_score
        FROM diagnostic_reports
        WHERE session_id = %s
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (session_id,),
        fetch="one",
    ) or {}

    flat = _flatten_context(base)
    age = _age_from_dob(base.get("date_of_birth"))

    fields_populated = dict(base.get("fields_populated") or {})
    weights = get_all_field_weights()
    missing_fields = [name for name in weights if not fields_populated.get(name)]

    completeness = _to_float(base.get("completeness_score"))
    status_badge = _status_badge(completeness)

    payload = {
        "session": {
            "session_id": str(base.get("session_id")),
            "patient_id": str(base.get("patient_id")),
            "status": str(base.get("status") or "active"),
            "created_at": _iso(base.get("created_at")),
            "first_name": str(base.get("first_name") or ""),
            "last_name": str(base.get("last_name") or ""),
            "full_name": f"{base.get('first_name') or ''} {base.get('last_name') or ''}".strip(),
            "mrn": str(base.get("mrn") or ""),
            "age": age,
            "sex": str(base.get("sex") or ""),
            "date_of_birth": _display_date(base.get("date_of_birth")),
        },
        "completeness": {
            "score": completeness,
            "percent": int(round(completeness * 100)),
            "missing_count": len(missing_fields),
            "status_badge": status_badge,
        },
        "patient_context": {
            "symptoms": dict(base.get("symptoms") or {}),
            "clinical": dict(base.get("clinical") or {}),
            "genomics": dict(base.get("genomics") or {}),
            "vasari": dict(base.get("vasari") or {}),
            "pathology": dict(base.get("pathology") or {}),
            "labs": dict(base.get("labs") or {}),
            "treatment_history": dict(base.get("treatment_history") or {}),
            "fields_populated": fields_populated,
        },
        "imaging_available": bool(imaging),
        "metrics": _build_metric_cards(imaging) if imaging else [],
        "key_fields": _build_key_fields(flat, age),
        "auto_flags": _build_auto_flags(flat, latest_report),
        "provisional_grade": _build_provisional_grade(flat, imaging, latest_report),
        "conversation": _load_conversation(session_id),
        "pending_question": None,
    }

    return jsonify(payload)


def _load_patient_records(query: str) -> list[dict[str, Any]]:
    search = query.strip()
    like = f"%{search}%"
    limit = 120 if search else 48

    rows = execute_query(
        """
        WITH latest_imaging AS (
            SELECT DISTINCT ON (session_id)
                   session_id,
                   annotated_dir,
                   wt_volume_cm3,
                   et_volume_cm3,
                   scan_date,
                   created_at
            FROM imaging_reports
            ORDER BY session_id, created_at DESC
        ),
        latest_report AS (
            SELECT DISTINCT ON (session_id)
                   session_id,
                   diagnosis_label,
                   who_grade_predicted,
                   confidence_score,
                   created_at
            FROM diagnostic_reports
            ORDER BY session_id, created_at DESC
        )
        SELECT p.id AS patient_id,
               p.first_name,
               p.last_name,
               p.mrn,
               p.date_of_birth,
               p.sex,
               s.id AS session_id,
               s.status,
               s.created_at AS session_created_at,
               pc.completeness_score,
               li.annotated_dir,
               li.wt_volume_cm3,
               li.et_volume_cm3,
               li.scan_date,
               lr.diagnosis_label,
               lr.who_grade_predicted,
               lr.confidence_score
        FROM sessions s
        JOIN patients p ON p.id = s.patient_id
        LEFT JOIN patient_context pc ON pc.session_id = s.id
        LEFT JOIN latest_imaging li ON li.session_id = s.id
        LEFT JOIN latest_report lr ON lr.session_id = s.id
        WHERE (
            %s = ''
            OR COALESCE(p.mrn, '') ILIKE %s
            OR CONCAT_WS(' ', COALESCE(p.first_name, ''), COALESCE(p.last_name, '')) ILIKE %s
            OR COALESCE(p.first_name, '') ILIKE %s
            OR COALESCE(p.last_name, '') ILIKE %s
        )
        ORDER BY s.created_at DESC, p.last_name ASC NULLS LAST, p.first_name ASC NULLS LAST
        LIMIT %s
        """,
        (search, like, like, like, like, limit),
        fetch="all",
    ) or []

    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        patient_id = str(row["patient_id"])
        if patient_id not in grouped:
            grouped[patient_id] = {
                "patient_id": patient_id,
                "first_name": str(row.get("first_name") or ""),
                "last_name": str(row.get("last_name") or ""),
                "full_name": f"{row.get('first_name') or ''} {row.get('last_name') or ''}".strip() or "Unnamed patient",
                "mrn": str(row.get("mrn") or ""),
                "age": _age_from_dob(row.get("date_of_birth")),
                "sex": str(row.get("sex") or ""),
                "sessions": [],
            }

        completeness = _to_float(row.get("completeness_score"), default=0.0)
        confidence = _to_float(row.get("confidence_score"), default=0.0)
        grouped[patient_id]["sessions"].append(
            {
                "session_id": str(row.get("session_id") or ""),
                "status": str(row.get("status") or "active"),
                "session_created_at": _display_datetime(row.get("session_created_at")),
                "completeness_percent": int(round(completeness * 100)),
                "diagnosis_label": str(row.get("diagnosis_label") or ""),
                "who_grade_predicted": str(row.get("who_grade_predicted") or ""),
                "confidence_score": round(confidence, 2),
                "scan_date": _display_date(row.get("scan_date")),
                "wt_volume_cm3": round(_to_float(row.get("wt_volume_cm3")), 2),
                "et_volume_cm3": round(_to_float(row.get("et_volume_cm3")), 2),
                "has_report": bool(row.get("diagnosis_label") or row.get("who_grade_predicted")),
                "has_annotated": bool(row.get("annotated_dir")),
            }
        )

    return list(grouped.values())


def _iso(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    return None


def _display_datetime(value: Any) -> str:
    if isinstance(value, datetime):
        return value.strftime("%d %b %Y, %I:%M %p")
    return "Unknown"


def _display_date(value: Any) -> str:
    if isinstance(value, datetime):
        return value.strftime("%d %b %Y")
    if isinstance(value, date):
        return value.strftime("%d %b %Y")
    if isinstance(value, str) and value:
        try:
            parsed = date.fromisoformat(value)
            return parsed.strftime("%d %b %Y")
        except ValueError:
            return value
    return ""


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "mutated",
            "methylated",
            "deleted",
            "present",
        }
    return False


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


def _flatten_context(row: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for section in [
        "symptoms",
        "clinical",
        "genomics",
        "vasari",
        "pathology",
        "labs",
        "treatment_history",
    ]:
        data = row.get(section)
        if isinstance(data, dict):
            flat.update(data)
    return flat


def _status_badge(completeness: float) -> dict[str, str]:
    if completeness < 0.60:
        return {"tone": "warn", "label": "Awaiting critical fields"}
    if completeness < 0.85:
        return {"tone": "info", "label": "Collecting clinical context"}
    return {"tone": "ok", "label": "Ready to grade"}


def _metric_severity(value: float, warn_at: float, danger_at: float) -> str:
    if value >= danger_at:
        return "danger"
    if value >= warn_at:
        return "warn"
    return "ok"


def _build_metric_cards(imaging: dict[str, Any]) -> list[dict[str, Any]]:
    wt = _to_float(imaging.get("wt_volume_cm3"))
    nec = _to_float(imaging.get("necrosis_ratio"))
    edema = _to_float(imaging.get("edema_ratio"))
    shift = _to_float(imaging.get("midline_shift_mm"))

    wt_sev = _metric_severity(wt, warn_at=40.0, danger_at=60.0)
    nec_sev = _metric_severity(nec, warn_at=0.20, danger_at=0.35)
    edema_sev = _metric_severity(edema, warn_at=1.0, danger_at=1.8)
    shift_sev = _metric_severity(shift, warn_at=3.0, danger_at=5.0)

    subtitle_map = {
        "wt": {
            "ok": "Tumor burden within range",
            "warn": "Large tumor burden",
            "danger": "Very large tumor burden",
        },
        "nec": {
            "ok": "No aggressive necrosis signal",
            "warn": "Elevated necrotic fraction",
            "danger": "Grade 4 threshold exceeded",
        },
        "edema": {
            "ok": "Edema burden limited",
            "warn": "Elevated edema burden",
            "danger": "Significant mass effect",
        },
        "shift": {
            "ok": "Within safe range",
            "warn": "Moderate shift",
            "danger": "Critical shift risk",
        },
    }

    return [
        {
            "id": "wt_volume",
            "label": "Whole tumor",
            "value": round(wt, 2),
            "unit": "cm3",
            "severity": wt_sev,
            "subtitle": subtitle_map["wt"][wt_sev],
        },
        {
            "id": "necrosis_ratio",
            "label": "Necrosis ratio",
            "value": round(nec, 3),
            "unit": "",
            "severity": nec_sev,
            "subtitle": subtitle_map["nec"][nec_sev],
        },
        {
            "id": "edema_ratio",
            "label": "Edema ratio",
            "value": round(edema, 3),
            "unit": "",
            "severity": edema_sev,
            "subtitle": subtitle_map["edema"][edema_sev],
        },
        {
            "id": "midline_shift",
            "label": "Midline shift",
            "value": round(shift, 2),
            "unit": "mm",
            "severity": shift_sev,
            "subtitle": subtitle_map["shift"][shift_sev],
        },
    ]


def _display_value(value: Any) -> str:
    if value is None:
        return "not tested"
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned if cleaned else "not tested"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value) if value else "not tested"
    return str(value)


def _build_key_fields(flat: dict[str, Any], age: int | None) -> list[dict[str, Any]]:
    keys = [
        ("symptom_duration_days", "Symptom duration"),
        ("seizure_onset_new", "New-onset seizure"),
        ("headache_pattern", "Headache pattern"),
        ("age_at_diagnosis", "Age at diagnosis"),
        ("idh_status", "IDH status"),
        ("mgmt_methylation", "MGMT methylation"),
        ("karnofsky_score", "KPS score"),
        ("enhancement_pattern", "Enhancement pattern"),
        ("necrosis_on_histology", "Necrosis on histology"),
        ("tert_promoter_mutation", "TERT promoter"),
    ]

    output: list[dict[str, Any]] = []
    for key, label in keys:
        raw = flat.get(key)
        if raw is None and key == "age_at_diagnosis" and age is not None:
            raw = age
        display = _display_value(raw)
        output.append(
            {
                "key": key,
                "label": label,
                "value": display,
                "missing": display == "not tested",
            }
        )
    return output


def _build_auto_flags(flat: dict[str, Any], report: dict[str, Any]) -> list[dict[str, str]]:
    flags: list[dict[str, str]] = []

    kps = _to_float(flat.get("karnofsky_score"), default=-1)
    if 0 <= kps < 70:
        flags.append(
            {
                "severity": "warning",
                "text": "KPS below 70. Consider palliative-priority pathway and trial exclusions.",
            }
        )

    if _to_bool(flat.get("eloquent_cortex_involvement")):
        flags.append(
            {
                "severity": "critical",
                "text": "Eloquent cortex involvement. Functional mapping guidance recommended.",
            }
        )

    if _to_bool(flat.get("ependymal_invasion")):
        flags.append(
            {
                "severity": "warning",
                "text": "Ependymal invasion detected. CSF dissemination workup advised.",
            }
        )

    if _to_bool(flat.get("prior_cranial_irradiation")):
        flags.append(
            {
                "severity": "critical",
                "text": "Prior cranial irradiation history. Re-irradiation dose constraints apply.",
            }
        )

    if _to_bool(flat.get("crosses_midline")):
        flags.append(
            {
                "severity": "warning",
                "text": "Midline crossing pattern detected. Gross total resection may be limited.",
            }
        )

    if _to_bool(flat.get("bevacizumab_use")):
        flags.append(
            {
                "severity": "info",
                "text": "Anti-VEGF therapy detected - ET volume interpretation may be affected.",
            }
        )

    grade = str(report.get("who_grade_predicted") or "").strip()
    mgmt = str(flat.get("mgmt_methylation") or "").strip().lower()
    if grade == "4" and mgmt == "methylated":
        flags.append(
            {
                "severity": "info",
                "text": "MGMT methylated in Grade 4 profile. TMZ response may be comparatively favorable.",
            }
        )

    return flags


def _build_provisional_grade(
    flat: dict[str, Any],
    imaging: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    # Prefer latest persisted report when available.
    report_grade = str(report.get("who_grade_predicted") or "").strip()
    report_conf = _to_float(report.get("confidence_score"), default=0.0)
    if report_grade:
        return {
            "label": f"WHO Grade {report_grade}",
            "confidence": round(report_conf, 3),
            "note": "Derived from latest generated diagnostic report.",
            "report_unlock_threshold": 0.75,
            "is_unlocked": report_conf >= 0.75,
        }

    necrosis_ratio = _to_float(imaging.get("necrosis_ratio"), default=0.0)
    enhancement_pattern = str(flat.get("enhancement_pattern") or "").lower()
    idh = str(flat.get("idh_status") or "").lower()
    h3k27m = str(flat.get("h3k27m_mutation") or "").lower()
    codeletion = str(flat.get("codeletion_1p19q") or "").lower()
    cdkn = str(flat.get("cdkn2a_b_deletion") or "").lower()

    grade = 3
    conf = 0.50
    note = "Provisional estimate while additional fields are collected."

    if "mutated" in h3k27m:
        grade = 4
        conf = 0.82
        note = "H3K27M-altered profile strongly supports Grade 4 behavior."
    elif "mutant" in idh and "deleted" in cdkn:
        grade = 4
        conf = 0.78
        note = "IDH-mutant with CDKN2A/B deletion indicates high-grade astrocytoma profile."
    elif "wild" in idh and (necrosis_ratio > 0.05 or "ring" in enhancement_pattern):
        grade = 4
        conf = 0.72
        note = "IDH-wildtype with necrosis/ring enhancement suggests glioblastoma pattern."
    elif "mutant" in idh and "codeleted" in codeletion:
        grade = 2
        conf = 0.63
        note = "IDH-mutant and 1p/19q-codeleted pattern leans oligodendroglioma spectrum."
    elif necrosis_ratio >= 0.30:
        grade = 4
        conf = 0.61
        note = "High necrosis fraction suggests aggressive phenotype pending molecular confirmation."
    elif necrosis_ratio >= 0.15:
        grade = 3
        conf = 0.56
        note = "Moderate necrotic burden suggests intermediate-to-high grade risk."
    else:
        grade = 2
        conf = 0.45
        note = "Insufficient high-risk markers; continue field completion for definitive grading." 

    return {
        "label": f"WHO Grade {grade} (provisional)",
        "confidence": round(conf, 3),
        "note": note,
        "report_unlock_threshold": 0.75,
        "is_unlocked": conf >= 0.75,
    }


def _load_conversation(session_id: str) -> list[dict[str, Any]]:
    rows = execute_query(
        """
        SELECT role, content, content_json, created_at
        FROM messages
        WHERE session_id = %s
        ORDER BY created_at ASC
        LIMIT 250
        """,
        (session_id,),
        fetch="all",
    ) or []

    out: list[dict[str, Any]] = []
    for row in rows:
        role = str(row.get("role") or "assistant")
        created_at = row.get("created_at")
        payload = row.get("content_json")

        if role == "assistant" and isinstance(payload, dict):
            out.append(
                {
                    "role": role,
                    "kind": "json",
                    "payload": payload,
                    "created_at": _iso(created_at),
                }
            )
        elif role == "tool" and isinstance(payload, dict):
            out.append(
                {
                    "role": role,
                    "kind": "tool",
                    "payload": payload,
                    "text": str(row.get("content") or ""),
                    "created_at": _iso(created_at),
                }
            )
        else:
            out.append(
                {
                    "role": role,
                    "kind": "text",
                    "text": str(row.get("content") or ""),
                    "created_at": _iso(created_at),
                }
            )

    return out


def _pending_question_payload(session_id: str) -> dict[str, Any] | None:
    row = execute_query(
        """
        SELECT question_id
        FROM questions_asked
        WHERE session_id = %s AND answered_at IS NULL
        ORDER BY id DESC
        LIMIT 1
        """,
        (session_id,),
        fetch="one",
    )

    if not row or not row.get("question_id"):
        return None

    qid = str(row["question_id"])
    msg = execute_query(
        """
        SELECT content_json
        FROM messages
        WHERE session_id = %s
          AND role = 'assistant'
          AND content_json->>'question_id' = %s
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (session_id, qid),
        fetch="one",
    )

    if not msg:
        return {"question_id": qid}

    payload = msg.get("content_json")
    return payload if isinstance(payload, dict) else {"question_id": qid}
