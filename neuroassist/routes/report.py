"""
routes/report.py

Report endpoints:
- GET /report/<session_id>        : standalone diagnostic report view
- GET /report/<session_id>/print  : print-optimized report stub (browser print to PDF)
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from flask import Blueprint, abort, current_app, render_template, url_for

from db.connection import execute_query

report_bp = Blueprint("report", __name__)

DISCLAIMER = (
    "NeuroAssist is a clinical decision support tool. All outputs must be reviewed by a "
    "qualified neuro-oncologist before any clinical action is taken."
)


@report_bp.route("/report/<session_id>")
def report_view(session_id: str):
    return _render_report(session_id=session_id, print_mode=False)


@report_bp.route("/report/<session_id>/print")
def report_print_view(session_id: str):
    return _render_report(session_id=session_id, print_mode=True)


@report_bp.route("/report/<session_id>/annotated")
def annotated_slices_view(session_id: str):
    sess = execute_query(
        """
        SELECT s.id AS session_id,
               p.first_name,
               p.last_name
        FROM sessions s
        JOIN patients p ON p.id = s.patient_id
        WHERE s.id = %s
        """,
        (session_id,),
        fetch="one",
    )
    if not sess:
        return "Session not found", 404

    imaging_row = execute_query(
        """
        SELECT annotated_dir
        FROM imaging_reports
        WHERE session_id = %s
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (session_id,),
        fetch="one",
    )
    if not imaging_row or not imaging_row.get("annotated_dir"):
        abort(404, description="Annotated slices not available for this session.")

    assets = _resolve_annotated_assets(imaging_row["annotated_dir"])
    if not assets:
        abort(404, description="Annotated slice files were not found on disk.")

    session = dict(sess)
    return render_template(
        "annotated_slices.html",
        session=session,
        summary_image=assets.get("summary_image", ""),
        axial_slices=assets.get("axial_slices", []),
    )


def _render_report(session_id: str, print_mode: bool):
    sess = execute_query(
        """
        SELECT s.id AS session_id,
               s.status,
               s.created_at,
               p.id AS patient_id,
               p.first_name,
               p.last_name,
               p.mrn,
               p.date_of_birth,
               p.sex
        FROM sessions s
        JOIN patients p ON p.id = s.patient_id
        WHERE s.id = %s
        """,
        (session_id,),
        fetch="one",
    )
    if not sess:
        return "Session not found", 404

    report_row = execute_query(
        """
        SELECT *
        FROM diagnostic_reports
        WHERE session_id = %s
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (session_id,),
        fetch="one",
    )

    imaging_row = execute_query(
        """
        SELECT *
        FROM imaging_reports
        WHERE session_id = %s
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (session_id,),
        fetch="one",
    )

    session = dict(sess)
    report = dict(report_row) if report_row else {}
    imaging = dict(imaging_row) if imaging_row else {}

    age = _age_from_dob(session.get("date_of_birth"))

    full = report.get("full_report") if isinstance(report.get("full_report"), dict) else {}
    survival_outlook = full.get("survival_outlook") if isinstance(full.get("survival_outlook"), dict) else {}

    if not report and full:
        report = {
            "who_grade_predicted": full.get("who_grade_predicted"),
            "diagnosis_label": full.get("diagnosis_label"),
            "confidence_score": full.get("confidence_score"),
            "data_completeness": full.get("data_completeness"),
            "treatment_flags": full.get("treatment_flags", []),
            "factors_favorable": survival_outlook.get("factors_favorable", []),
            "factors_unfavorable": survival_outlook.get("factors_unfavorable", []),
            "survival_category": survival_outlook.get("category"),
            "survival_score": survival_outlook.get("score"),
            "estimated_median_months": survival_outlook.get("estimated_median_months"),
            "recommendation_summary": full.get("recommendation_summary"),
            "disclaimer": full.get("disclaimer"),
        }

    if report and "disclaimer" not in report:
        report["disclaimer"] = DISCLAIMER

    if report and not report.get("treatment_flags") and isinstance(full.get("treatment_flags"), list):
        report["treatment_flags"] = full.get("treatment_flags")

    if report and not report.get("factors_favorable") and isinstance(survival_outlook.get("factors_favorable"), list):
        report["factors_favorable"] = survival_outlook.get("factors_favorable")

    if report and not report.get("factors_unfavorable") and isinstance(survival_outlook.get("factors_unfavorable"), list):
        report["factors_unfavorable"] = survival_outlook.get("factors_unfavorable")

    if report and not report.get("estimated_median_months"):
        report["estimated_median_months"] = survival_outlook.get("estimated_median_months")

    if report and not report.get("report_text"):
        report["report_text"] = full.get("report_text")

    if report and not report.get("recommendation_summary"):
        report["recommendation_summary"] = full.get("recommendation_summary")

    return render_template(
        "report.html",
        session=session,
        report=report,
        imaging=imaging,
        patient_age=age,
        print_mode=print_mode,
        disclaimer=DISCLAIMER,
    )


def _resolve_annotated_assets(stored_path: str) -> dict[str, list[str] | str] | None:
    if not stored_path:
        return None

    candidate = _resolve_stored_path(stored_path)
    if candidate is None:
        return None

    annotated_dir = candidate if candidate.is_dir() else candidate.parent
    if not annotated_dir.exists():
        return None

    summary_path = annotated_dir / "summary_3plane.png"
    axial_paths = sorted(annotated_dir.glob("axial_z*.png"))

    summary_image = _static_url_for_path(summary_path) if summary_path.exists() else ""
    axial_slices = [url for path in axial_paths if (url := _static_url_for_path(path))]

    if not summary_image and not axial_slices:
        return None

    return {
        "summary_image": summary_image,
        "axial_slices": axial_slices,
    }


def _resolve_stored_path(stored_path: str) -> Path | None:
    value = str(stored_path).strip()
    if not value:
        return None

    static_root = Path(current_app.static_folder).resolve()

    if value.startswith("/static/"):
        relative = value[len("/static/") :]
        return (static_root / relative).resolve()

    path = Path(value)
    if path.is_absolute():
        return path.resolve()

    return (static_root / value.lstrip("/")).resolve()


def _static_url_for_path(path: Path) -> str:
    try:
        relative = path.resolve().relative_to(Path(current_app.static_folder).resolve())
    except ValueError:
        return ""

    return url_for("static", filename=str(relative).replace("\\", "/"))


def _age_from_dob(dob: Any) -> int | None:
    if not dob:
        return None

    if isinstance(dob, date):
        today = date.today()
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

    if isinstance(dob, str):
        try:
            parsed = date.fromisoformat(dob)
            today = date.today()
            return today.year - parsed.year - ((today.month, today.day) < (parsed.month, parsed.day))
        except ValueError:
            return None

    return None
