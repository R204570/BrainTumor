"""
services/completeness.py
────────────────────────
Computes the weighted completeness score for a patient diagnostic context.

IMPORTANT: This is the AUTHORITATIVE scorer — the LLM never computes this.
Called after each structured clinical intake submission. Result stored in patient_context.completeness_score.
"""

from db.connection import execute_query

SECTION_FIELDS = {
    "symptoms": {
        "symptom_duration_days",
        "seizure_onset_new",
        "headache_pattern",
        "headache_present",
        "nausea_vomiting",
        "cognitive_decline",
        "memory_deficit",
        "personality_change",
        "visual_field_deficit",
    },
    "clinical": {
        "age_at_diagnosis",
        "karnofsky_score",
        "speech_deficit",
        "motor_deficit",
        "visual_deficit",
        "mmse_score",
        "sex",
        "ecog_score",
        "comorbidities",
        "extent_of_resection",
    },
    "genomics": {
        "idh_status",
        "mgmt_methylation",
        "tert_promoter_mutation",
        "cdkn2a_b_deletion",
        "codeletion_1p19q",
        "egfr_amplification",
        "h3k27m_mutation",
    },
    "vasari": {
        "enhancement_pattern",
        "ependymal_invasion",
        "multifocal",
        "crosses_midline",
        "eloquent_cortex_involvement",
        "ring_enhancing",
        "necrosis_proportion_imaging",
        "dce_cbv_ratio",
        "adc_min_value",
    },
    "pathology": {
        "necrosis_on_histology",
        "ki67_index_percent",
        "mitotic_index",
        "histology_type",
        "microvascular_proliferation",
    },
    "labs": {
        "ldh_level",
        "steroid_dose_mg",
        "csf_protein_elevated",
    },
    "treatment_history": {
        "prior_cranial_irradiation",
        "bevacizumab_use",
        "prior_temozolomide",
        "prior_surgery_type",
    },
}

# ─── Field weight tiers (from project spec / neuroassist_full_spec.html) ─────

CRITICAL_FIELDS = {
    # weight = 3 each
    "symptom_duration_days",
    "seizure_onset_new",
    "headache_pattern",
    "age_at_diagnosis",
    "idh_status",
    "mgmt_methylation",
    "karnofsky_score",
    "enhancement_pattern",
    "necrosis_on_histology",
    "tert_promoter_mutation",
}

HIGH_FIELDS = {
    # weight = 2 each
    "extent_of_resection",
    "speech_deficit",
    "motor_deficit",
    "visual_deficit",
    "cdkn2a_b_deletion",
    "ependymal_invasion",
    "mmse_score",
}

# All remaining ~28 tracked fields  — weight = 1 each
MEDIUM_LOW_FIELDS = {
    # Symptoms
    "headache_present", "nausea_vomiting", "cognitive_decline",
    "memory_deficit", "personality_change", "visual_field_deficit",
    # Clinical
    "sex", "ecog_score", "comorbidities",
    # Genomics
    "codeletion_1p19q", "egfr_amplification", "h3k27m_mutation",
    "ki67_index_percent", "mitotic_index",
    # VASARI
    "multifocal", "crosses_midline", "eloquent_cortex_involvement",
    "ring_enhancing", "necrosis_proportion_imaging",
    "dce_cbv_ratio", "adc_min_value",
    # Pathology
    "histology_type", "microvascular_proliferation",
    # Labs
    "ldh_level", "steroid_dose_mg", "csf_protein_elevated",
    # Treatment history
    "prior_cranial_irradiation", "bevacizumab_use",
    "prior_temozolomide", "prior_surgery_type",
}

# Computed totals (cached for speed)
_WEIGHT_MAP: dict[str, int] = {}
_TOTAL_WEIGHT: int = 0
_FIELD_SECTION_MAP: dict[str, str] = {}


def _build_weight_map() -> None:
    """Populate the weight map once at module load."""
    global _WEIGHT_MAP, _TOTAL_WEIGHT, _FIELD_SECTION_MAP
    for f in CRITICAL_FIELDS:
        _WEIGHT_MAP[f] = 3
    for f in HIGH_FIELDS:
        _WEIGHT_MAP[f] = 2
    for f in MEDIUM_LOW_FIELDS:
        _WEIGHT_MAP[f] = 1

    for section, fields in SECTION_FIELDS.items():
        for field in fields:
            _FIELD_SECTION_MAP[field] = section

    _TOTAL_WEIGHT = sum(_WEIGHT_MAP.values())


_build_weight_map()


def compute_completeness_score(fields_populated: dict) -> tuple[float, list[str]]:
    """
    Pure function — compute the completeness score from a fields_populated dict.

    Parameters
    ----------
    fields_populated : dict  {field_name: True/False}
        Anything not present is treated as False (unpopulated).

    Returns
    -------
    (score, missing_fields)
        score         : float 0.0 – 1.0
        missing_fields: list of field names not yet populated, sorted by weight desc
    """
    earned = 0
    missing = []

    for field, weight in _WEIGHT_MAP.items():
        if fields_populated.get(field):
            earned += weight
        else:
            missing.append((field, weight))

    score = round(earned / _TOTAL_WEIGHT, 3) if _TOTAL_WEIGHT > 0 else 0.0

    # Sort missing by weight descending (highest priority first)
    missing.sort(key=lambda x: x[1], reverse=True)
    missing_fields = [f for f, _ in missing]

    return score, missing_fields


def update_completeness_in_db(
    patient_id: str,
    session_id: str,
    fields_populated: dict,
) -> tuple[float, list[str]]:
    """
    Compute completeness score and persist it to patient_context.

    Returns
    -------
    (score, missing_fields)
    """
    score, missing_fields = compute_completeness_score(fields_populated)

    execute_query(
        """
        UPDATE patient_context
        SET completeness_score = %s,
            updated_at         = NOW()
        WHERE patient_id = %s AND session_id = %s
        """,
        (score, patient_id, session_id),
    )

    return score, missing_fields


def get_fields_populated(patient_id: str, session_id: str) -> dict:
    """
    Load the fields_populated map from patient_context.

    Returns
    -------
    dict {field_name: bool}  — empty dict if no context row exists
    """
    row = execute_query(
        """
        SELECT fields_populated
        FROM patient_context
        WHERE patient_id = %s AND session_id = %s
        """,
        (patient_id, session_id),
        fetch="one",
    )
    if row and row.get("fields_populated"):
        return dict(row["fields_populated"])
    return {}


def get_all_field_weights() -> dict[str, int]:
    """Return the full field→weight mapping (for introspection / UI display)."""
    return dict(_WEIGHT_MAP)


def get_field_section_map() -> dict[str, str]:
    """Return the canonical field→section mapping used for patient_context updates."""
    return dict(_FIELD_SECTION_MAP)


def get_allowed_fields_by_section() -> dict[str, list[str]]:
    """Return the allowed interview fields grouped by patient_context section."""
    return {
        section: sorted(fields)
        for section, fields in SECTION_FIELDS.items()
    }
