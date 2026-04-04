"""
rag_prompts.py

Prompt builders for NeuroAssist final-report generation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


AI_DISCLAIMER = (
    "This is an AI-assisted analysis. Final diagnosis requires multidisciplinary "
    "clinical review and histopathological confirmation."
)


SYSTEM_PROMPT = """
You are NeuroAssist, an AI-powered neuro-oncology diagnostic assistant integrated
into a clinical decision-support workflow for brain tumor evaluation.

You reason over:
- segmentation outputs from a 3D Attention U-Net trained on BraTS-style MRI data
- structured patient context captured during the clinician interview
- retrieved neuro-oncology RAG snippets supplied in the prompt

Segmentation labels:
- Label 0 = Background
- Label 1 = NCR/NET (Necrotic Core / Non-Enhancing Tumor)
- Label 2 = ED (Peritumoral Edema)
- Label 3 = ET (Enhancing Tumor)

Composite regions:
- WT = Whole Tumor = Labels 1 + 2 + 3
- TC = Tumor Core = Labels 1 + 3
- ET = Enhancing Tumor = Label 3 only

Clinical knowledge to apply:
- WHO CNS Tumor Classification 2021 principles
- IDH mutation, MGMT methylation, 1p/19q status, EGFR and TERT implications
- Imaging hallmarks of glioblastoma and lower-grade glioma
- Radiomics interpretation using ET/WT, NCR/TC, edema burden, diameter, and location
- Population-level survival data only, never patient-specific prognosis
- Multidisciplinary treatment planning principles for surgery, radiation, and systemic therapy

Style:
- Clinical, measured, and precise
- Explicit about uncertainty and missing data
- Every major inference must reference the imaging metric, molecular marker, or clinical detail that supports it

Hard rules:
1. Do not give a definitive diagnosis; provide probability-weighted impressions.
2. Do not recommend specific drug doses.
3. If a required clinical or molecular indicator is absent, mark it as UNKNOWN or Not provided.
4. Survival language must stay population-level and non-deterministic.
5. Always include this sentence verbatim in the final report disclaimer:
   "This is an AI-assisted analysis. Final diagnosis requires multidisciplinary clinical review and histopathological confirmation."
""".strip()


def build_system_prompt() -> str:
    """Return the shared base system prompt."""
    return SYSTEM_PROMPT


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _format_unknown(value: Any, *, empty_text: str = "UNKNOWN") -> str:
    if value is None:
        return empty_text
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned if cleaned else empty_text
    if isinstance(value, (list, tuple)):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return ", ".join(cleaned) if cleaned else empty_text
    return str(value)


def _format_conversation_history(conversation_history: list[dict[str, str]]) -> str:
    if not conversation_history:
        return "[No conversation history yet]"

    lines: list[str] = []
    for turn in conversation_history:
        role = str(turn.get("role") or "").strip().lower()
        speaker = "Clinician" if role == "user" else "NeuroAssist"
        content = str(turn.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines) if lines else "[No conversation history yet]"


def build_context_block(
    metrics: dict[str, Any],
    lobe_inv: dict[str, Any],
    patient_id: str = "Unknown",
    *,
    derived_metrics: dict[str, Any] | None = None,
    missing_fields: list[str] | None = None,
    scan_filename: str = "",
) -> str:
    """
    Build the imaging-heavy RAG context block shared by the interview and report prompts.
    """
    derived_metrics = derived_metrics or {}
    missing_fields = missing_fields or []

    wt = _safe_float(metrics.get("wt_volume_cm3"))
    ncr = _safe_float(metrics.get("ncr_volume_cm3"))
    ed = _safe_float(metrics.get("ed_volume_cm3"))
    et = _safe_float(metrics.get("et_volume_cm3"))
    diameter = _safe_float(metrics.get("diameter_mm"))
    centroid = metrics.get("centroid") or (64, 64, 64)
    cz, cy, cx = (
        _safe_int(centroid[0] if len(centroid) > 0 else 64, 64),
        _safe_int(centroid[1] if len(centroid) > 1 else 64, 64),
        _safe_int(centroid[2] if len(centroid) > 2 else 64, 64),
    )

    tc = ncr + et
    et_wt_ratio = round(et / wt, 3) if wt > 0 else 0.0
    ncr_tc_ratio = round(ncr / tc, 3) if tc > 0 else 0.0
    edema_wt_ratio = round(ed / wt, 3) if wt > 0 else 0.0
    enhancement_ratio = _safe_float(derived_metrics.get("enhancement_ratio"), et_wt_ratio)
    necrosis_ratio = _safe_float(derived_metrics.get("necrosis_ratio"), ncr_tc_ratio)
    edema_ratio = _safe_float(derived_metrics.get("edema_ratio"), edema_wt_ratio)
    midline_shift_mm = _safe_float(derived_metrics.get("midline_shift_mm"))

    dominant_lobe = "Unknown"
    if lobe_inv:
        dominant_lobe = max(lobe_inv, key=lambda key: _safe_float(lobe_inv.get(key)))

    lobe_lines = [
        f"  - {lobe}: {_safe_float(pct):.1f}%"
        for lobe, pct in sorted(
            (lobe_inv or {}).items(),
            key=lambda item: _safe_float(item[1]),
            reverse=True,
        )
    ] or ["  - No lobar distribution available"]

    auto_flags: list[str] = []
    if wt > 50.0:
        auto_flags.append("High tumor burden: WT > 50 cm3 suggests elevated surgical complexity.")
    if et > 0 and et_wt_ratio > 0.25:
        auto_flags.append("High-grade imaging signal: ET present with ET/WT ratio above 0.25.")
    if ncr > 5.0 or ncr_tc_ratio > 0.50:
        auto_flags.append("Necrosis-heavy phenotype: NCR burden favors aggressive biology.")
    if edema_wt_ratio > 0.40:
        auto_flags.append("Infiltrative edema pattern: ED/WT ratio above 0.40.")
    if midline_shift_mm >= 5.0:
        auto_flags.append("Mass effect concern: midline shift is at least 5 mm.")
    if dominant_lobe in {"Frontal", "Temporal"}:
        auto_flags.append(
            "Dominant involvement in frontal/temporal territory may increase eloquent cortex risk."
        )
    if not auto_flags:
        auto_flags.append("No major automated high-risk imaging flag triggered from current thresholds.")

    missing_text = ", ".join(missing_fields[:12]) if missing_fields else "None currently tracked"

    return (
        f"=== IMAGING CONTEXT FOR PATIENT {patient_id} ===\n"
        f"Scan / source: {_format_unknown(scan_filename, empty_text='Not provided')}\n"
        f"Whole Tumor (WT): {wt:.2f} cm3\n"
        f"Tumor Core (TC): {tc:.2f} cm3\n"
        f"Necrotic Core (NCR): {ncr:.2f} cm3\n"
        f"Edema (ED): {ed:.2f} cm3\n"
        f"Enhancing Tumor (ET): {et:.2f} cm3\n"
        f"Maximum Diameter: {diameter:.1f} mm\n"
        f"Centroid (z, y, x): ({cz}, {cy}, {cx})\n"
        f"Dominant Lobe: {dominant_lobe}\n"
        f"Enhancement ratio: {enhancement_ratio:.3f}\n"
        f"Necrosis ratio: {necrosis_ratio:.3f}\n"
        f"Edema ratio: {edema_ratio:.3f}\n"
        f"ET/WT ratio: {et_wt_ratio:.3f}\n"
        f"NCR/TC ratio: {ncr_tc_ratio:.3f}\n"
        f"Midline shift: {midline_shift_mm:.2f} mm\n"
        "Lobe involvement:\n"
        + "\n".join(lobe_lines)
        + "\nAutomated imaging flags:\n- "
        + "\n- ".join(auto_flags)
        + f"\nHighest-priority missing tracked fields: {missing_text}\n"
        "=== END IMAGING CONTEXT ==="
    )


def build_report_prompt(
    context_block: str,
    conversation_history: list[dict[str, str]],
    patient_id: str = "Unknown",
    *,
    compact: bool = False,
) -> str:
    """
    Build the final structured-report prompt.
    """
    history_text = _format_conversation_history(conversation_history)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    narrative_instruction = (
        "Detailed narrative overview in multiple clinically readable paragraphs."
        if not compact
        else "Clinically readable narrative summary in 2 to 4 compact paragraphs."
    )
    detail_rule = (
        "- Be detailed, but keep each field concise enough for a local 8B model to complete reliably."
        if not compact
        else "- Keep every field compact and high-yield so the full JSON can complete reliably on a local 7B/8B model."
    )
    differential_rule = (
        "- Provide 3 ranked differential entries when feasible from the data, otherwise provide as many as the data supports."
    )

    return f"""
FINAL REPORT MODE

Context:
{context_block}

Clinician transcript:
{history_text}

Generate one detailed neuro-oncology report from the available intake and imaging data.
Return JSON only in this exact top-level shape:
{{
  "mode": "final_report",
  "report_text": "{narrative_instruction}",
  "who_grade_predicted": "Grade 1 | Grade 2 | Grade 3 | Grade 4 | Indeterminate",
  "diagnosis_label": "Primary probability-weighted diagnostic impression",
  "confidence_score": 0.0,
  "survival_category": "short category label",
  "survival_score": 0,
  "estimated_median_months": "population-level text estimate",
  "factors_favorable": ["factor"],
  "factors_unfavorable": ["factor"],
  "treatment_flags": [
    {{"severity": "info | warning | critical", "text": "flag text"}}
  ],
  "recommendation_summary": "Short MDT-oriented action summary.",
  "disclaimer": "{AI_DISCLAIMER}",
  "structured_report": {{
      "header": {{
        "patient_case_id": "{patient_id}",
        "report_generated": "{generated_at}",
        "ai_model": "3D Attention U-Net | BraTS-style segmentation + Groq LLM reasoning",
        "report_version": "2.0"
      }},
    "patient_clinical_summary": {{
      "demographics": "",
      "presenting_symptoms": "",
      "relevant_history": "",
      "current_medications": ""
    }},
    "imaging_findings": {{
      "tumor_volumes": [],
      "morphology": "",
      "localization": "",
      "radiomics_flags": []
    }},
    "differential_diagnosis": [
      {{
        "diagnosis": "",
        "probability": "High | Moderate | Low",
        "supporting": [],
        "against": []
      }}
    ],
    "molecular_profile_integration": {{
      "idh_status": "",
      "mgmt_methylation": "",
      "codeletion_1p19q": "",
      "other_markers": "",
      "molecular_grade": ""
    }},
    "grade_assessment": {{
      "likely_who_grade": "",
      "rationale": [],
      "confidence": "High | Moderate | Low",
      "confidence_explanation": ""
    }},
    "survival_estimate": {{
      "reference_class": "",
      "median_os": "",
      "two_year_os": "",
      "favorable_factors": [],
      "adverse_factors": [],
      "literature_basis": []
    }},
    "treatment_considerations": {{
      "surgical": "",
      "radiation": "",
      "chemotherapy": "",
      "monitoring": "",
      "mdt_referral": ""
    }},
    "uncertainty_limitations": {{
      "model_limitations": [],
      "missing_data": [],
      "recommended_tests": []
    }},
    "summary_for_mdt": []
  }}
}}

Mandatory report rules:
{detail_rule}
- Every major inference must cite the metric, marker, or clinical detail driving it.
- Use "Not provided" or "UNKNOWN" for missing data.
{differential_rule}
- Survival estimates must stay population-level.
- No markdown fences and no prose outside the JSON object.
""".strip()
