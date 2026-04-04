"""
services/completeness.py

Computes the weighted completeness score for the structured clinical context.
"""

from __future__ import annotations

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

CRITICAL_FIELDS = {
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
    "extent_of_resection",
    "speech_deficit",
    "motor_deficit",
    "visual_deficit",
    "cdkn2a_b_deletion",
    "ependymal_invasion",
    "mmse_score",
}

MEDIUM_LOW_FIELDS = {
    "headache_present",
    "nausea_vomiting",
    "cognitive_decline",
    "memory_deficit",
    "personality_change",
    "visual_field_deficit",
    "sex",
    "ecog_score",
    "comorbidities",
    "codeletion_1p19q",
    "egfr_amplification",
    "h3k27m_mutation",
    "ki67_index_percent",
    "mitotic_index",
    "multifocal",
    "crosses_midline",
    "eloquent_cortex_involvement",
    "ring_enhancing",
    "necrosis_proportion_imaging",
    "dce_cbv_ratio",
    "adc_min_value",
    "histology_type",
    "microvascular_proliferation",
    "ldh_level",
    "steroid_dose_mg",
    "csf_protein_elevated",
    "prior_cranial_irradiation",
    "bevacizumab_use",
    "prior_temozolomide",
    "prior_surgery_type",
}

_WEIGHT_MAP: dict[str, int] = {}
_TOTAL_WEIGHT: int = 0


def _build_weight_map() -> None:
    global _WEIGHT_MAP, _TOTAL_WEIGHT

    for field in CRITICAL_FIELDS:
        _WEIGHT_MAP[field] = 3
    for field in HIGH_FIELDS:
        _WEIGHT_MAP[field] = 2
    for field in MEDIUM_LOW_FIELDS:
        _WEIGHT_MAP[field] = 1

    _TOTAL_WEIGHT = sum(_WEIGHT_MAP.values())


_build_weight_map()


def compute_completeness_score(fields_populated: dict) -> tuple[float, list[str]]:
    earned = 0
    missing: list[tuple[str, int]] = []

    for field, weight in _WEIGHT_MAP.items():
        if fields_populated.get(field):
            earned += weight
        else:
            missing.append((field, weight))

    score = round(earned / _TOTAL_WEIGHT, 3) if _TOTAL_WEIGHT > 0 else 0.0
    missing.sort(key=lambda item: item[1], reverse=True)
    return score, [field for field, _ in missing]


def update_completeness_in_db(
    patient_id: str,
    session_id: str,
    fields_populated: dict,
) -> tuple[float, list[str]]:
    score, missing_fields = compute_completeness_score(fields_populated)

    execute_query(
        """
        UPDATE patient_context
        SET completeness_score = %s,
            updated_at = NOW()
        WHERE patient_id = %s AND session_id = %s
        """,
        (score, patient_id, session_id),
    )

    return score, missing_fields


def get_all_field_weights() -> dict[str, int]:
    """Return the field-to-weight mapping for the UI."""
    return dict(_WEIGHT_MAP)
