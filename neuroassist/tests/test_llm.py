"""
Unit tests for services/llm.py and final report context building.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from services.context_builder import build_input_context
from services.llm import Config, LLMServiceError, REPORT_SHORT_TEXT_LIMIT, _validate_report_response, call_final_report_model


def _mock_http_response(content: str):
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": content,
                }
            }
        ]
    }
    return response


class TestGroqLLM(unittest.TestCase):
    def test_call_final_report_model_with_markdown_wrapped_json(self):
        mock_response = {
            "mode": "final_report",
            "report_text": "Imaging and history together support a high-grade glioma profile.",
            "who_grade_predicted": "Grade IV",
            "diagnosis_label": "Glioblastoma",
            "confidence_score": 0.92,
            "survival_category": "guarded",
            "survival_score": 62,
            "estimated_median_months": "12-18 months",
            "factors_favorable": ["MGMT methylated"],
            "factors_unfavorable": ["ring enhancement"],
            "treatment_flags": [{"severity": "warning", "text": "Need tissue confirmation"}],
            "recommendation_summary": "Discuss tumor board review.",
            "disclaimer": "Clinical review required.",
            "structured_report": {
                "header": {
                    "patient_case_id": "CASE-1",
                    "report_generated": "2026-04-04 10:30",
                    "ai_model": "NeuroAssist",
                    "report_version": "2.0",
                },
                "patient_clinical_summary": {
                    "demographics": "52-year-old patient.",
                    "presenting_symptoms": "Headache.",
                    "relevant_history": "No prior tumor history.",
                    "current_medications": "Not provided",
                },
                "imaging_findings": {
                    "tumor_volumes": ["Whole Tumor (WT): 10 cm3"],
                    "morphology": "Ring-enhancing mass.",
                    "localization": "Frontal lobe.",
                    "radiomics_flags": ["ET present"],
                },
                "differential_diagnosis": [
                    {
                        "diagnosis": "Glioblastoma",
                        "probability": "High",
                        "supporting": ["Ring enhancement"],
                        "against": ["MGMT pending"],
                    }
                ],
                "molecular_profile_integration": {
                    "idh_status": "Not provided",
                    "mgmt_methylation": "Not provided",
                    "codeletion_1p19q": "Not provided",
                    "other_markers": "Not provided",
                    "molecular_grade": "Grade 4 favored",
                },
                "grade_assessment": {
                    "likely_who_grade": "Grade 4",
                    "rationale": ["Enhancement and necrosis support Grade 4"],
                    "confidence": "High",
                    "confidence_explanation": "Imaging profile is classic",
                },
                "survival_estimate": {
                    "reference_class": "GBM population",
                    "median_os": "12-18 months",
                    "two_year_os": "~25%",
                    "favorable_factors": ["MGMT methylated"],
                    "adverse_factors": ["Ring enhancement"],
                    "literature_basis": ["Stupp protocol"],
                },
                "treatment_considerations": {
                    "surgical": "Neurosurgical review advised",
                    "radiation": "Standard chemoradiation pathway if confirmed",
                    "chemotherapy": "TMZ suitability depends on pathology",
                    "monitoring": "Early MRI follow-up",
                    "mdt_referral": "Tumor board",
                },
                "uncertainty_limitations": {
                    "model_limitations": ["Segmentation-derived inference only"],
                    "missing_data": ["IDH unknown"],
                    "recommended_tests": ["Tissue diagnosis"],
                },
                "summary_for_mdt": ["High-grade glial neoplasm favored."],
            },
        }
        wrapped = f"```json\n{json.dumps(mock_response)}\n```"

        with patch.object(Config, "GROQ_API_KEY", "test-key"), patch("services.llm.httpx.post", return_value=_mock_http_response(wrapped)):
            result = call_final_report_model(
                input_context="- Patient ID: CASE-1",
                conversation_messages=[{"role": "user", "content": "Clinical intake submitted."}],
            )

        self.assertEqual(result["mode"], "final_report")
        self.assertEqual(result["who_grade_predicted"], "4")
        self.assertEqual(result["diagnosis_label"], "Glioblastoma")
        self.assertIn("structured_report", result)
        self.assertEqual(result["structured_report"]["header"]["patient_case_id"], "CASE-1")
        self.assertEqual(result["structured_report"]["grade_assessment"]["confidence"], "High")

    def test_call_final_report_model_retries_with_compact_prompt(self):
        mock_response = {
            "mode": "final_report",
            "report_text": "Compact but valid final report.",
            "who_grade_predicted": "Grade III",
            "diagnosis_label": "High-grade glioma",
            "confidence_score": 0.81,
            "survival_category": "intermediate",
            "survival_score": 55,
            "estimated_median_months": "18-24 months",
            "factors_favorable": ["Younger age"],
            "factors_unfavorable": ["Enhancement present"],
            "treatment_flags": [{"severity": "info", "text": "Histology still required"}],
            "recommendation_summary": "Proceed to MDT correlation.",
            "disclaimer": "Clinical review required.",
            "structured_report": {
                "header": {"patient_case_id": "CASE-2"},
                "patient_clinical_summary": {},
                "imaging_findings": {},
                "differential_diagnosis": [{"diagnosis": "High-grade glioma"}],
                "molecular_profile_integration": {},
                "grade_assessment": {},
                "survival_estimate": {},
                "treatment_considerations": {},
                "uncertainty_limitations": {},
                "summary_for_mdt": ["Escalate to tumor board."],
            },
        }

        call_counter = {"count": 0}

        def fake_post(*args, **kwargs):
            call_counter["count"] += 1
            if call_counter["count"] == 1:
                return _mock_http_response("not-json")
            return _mock_http_response(json.dumps(mock_response))

        with patch.object(Config, "GROQ_API_KEY", "test-key"), patch("services.llm.httpx.post", side_effect=fake_post):
            result = call_final_report_model(
                input_context="- Patient ID: CASE-2",
                conversation_messages=[{"role": "user", "content": "Clinical intake submitted."}],
            )

        self.assertEqual(call_counter["count"], 2)
        self.assertEqual(result["diagnosis_label"], "High-grade glioma")

    def test_call_final_report_model_with_invalid_json_raises(self):
        with patch.object(Config, "GROQ_API_KEY", "test-key"), patch("services.llm.httpx.post", return_value=_mock_http_response("This is not JSON")):
            with self.assertRaises(LLMServiceError):
                call_final_report_model(
                    input_context="- Patient ID: CASE-3",
                    conversation_messages=[],
                )

    def test_validate_report_response_clips_short_report_fields(self):
        long_text = "x" * (REPORT_SHORT_TEXT_LIMIT + 37)
        result = _validate_report_response(
            {
                "mode": "final_report",
                "report_text": "Valid report text.",
                "who_grade_predicted": "Grade 4",
                "diagnosis_label": "Glioblastoma",
                "confidence_score": 0.95,
                "survival_category": long_text,
                "survival_score": 70,
                "estimated_median_months": long_text,
                "factors_favorable": [],
                "factors_unfavorable": [],
                "treatment_flags": [],
            }
        )

        self.assertEqual(len(result["survival_category"]), REPORT_SHORT_TEXT_LIMIT)
        self.assertEqual(len(result["estimated_median_months"]), REPORT_SHORT_TEXT_LIMIT)

    def test_build_input_context_includes_registered_intake_profile(self):
        def fake_query(query, params=None, fetch=None):
            normalized = " ".join(str(query).split())
            if "FROM patient_context" in normalized:
                return {
                    "symptoms": {},
                    "clinical": {"sex": "Male", "age_at_diagnosis": 58},
                    "genomics": {},
                    "vasari": {},
                    "pathology": {},
                    "labs": {},
                    "treatment_history": {},
                    "fields_populated": {"sex": True, "age_at_diagnosis": True},
                    "completeness_score": 0.2,
                }
            if "FROM sessions s JOIN patients p" in normalized:
                return {
                    "first_name": "John",
                    "last_name": "Doe",
                    "mrn": "MRN-123",
                    "date_of_birth": "1968-01-01",
                    "sex": "Male",
                    "created_at": None,
                }
            if "FROM imaging_reports" in normalized:
                return {
                    "wt_volume_cm3": 12.3,
                    "ncr_volume_cm3": 4.2,
                    "ed_volume_cm3": 6.1,
                    "et_volume_cm3": 2.0,
                    "diameter_mm": 31.0,
                    "centroid_x": 60,
                    "centroid_y": 62,
                    "centroid_z": 58,
                    "lobe_pct_frontal": 70.0,
                    "lobe_pct_temporal": 20.0,
                    "lobe_pct_parietal": 10.0,
                    "lobe_pct_occipital": 0.0,
                    "lobe_pct_other": 0.0,
                    "necrosis_ratio": 0.25,
                    "enhancement_ratio": 0.16,
                    "edema_ratio": 0.50,
                    "midline_shift_mm": 0.0,
                }
            if "FROM messages" in normalized:
                return []
            return {}

        with patch("services.context_builder.execute_query", side_effect=fake_query):
            result = build_input_context(
                patient_id="patient-1",
                session_id="session-1",
                rag_chunks=[],
                missing_fields=["karnofsky_score", "idh_status"],
                current_user_message="",
                include_message_history=False,
            )

        text = result["input_context_str"]
        self.assertIn("Registered intake profile", text)
        self.assertIn("Patient name: John Doe", text)
        self.assertIn("MRN: MRN-123", text)
        self.assertIn("Biological sex: Male", text)


if __name__ == "__main__":
    unittest.main()
