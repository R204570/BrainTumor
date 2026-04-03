"""
Unit tests for services/llm.py.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from services.llm import (
    LLMResponseError,
    call_final_report_model,
    call_interview_model,
)


def _mock_http_response(content: str):
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "message": {
            "content": content,
        }
    }
    return response


class TestOllamaLLM(unittest.TestCase):
    def test_call_interview_model_with_valid_json_response(self):
        mock_response = {
            "mode": "interview",
            "assistant_message": "What is the patient's age and when did symptoms start?",
            "covered_fields": ["headache_present"],
            "context_updates": {
                "symptoms": {
                    "headache_present": True,
                }
            },
            "next_topics": ["age", "symptom duration"],
            "ready_for_report": False,
        }

        with patch("services.llm.httpx.post", return_value=_mock_http_response(json.dumps(mock_response))):
            result = call_interview_model(
                input_context="Dynamic context",
                conversation_messages=[{"role": "user", "content": "The patient has headaches."}],
                max_question_turns=15,
            )

        self.assertEqual(result["mode"], "interview")
        self.assertEqual(result["assistant_message"], mock_response["assistant_message"])
        self.assertEqual(result["covered_fields"], ["headache_present"])

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
        }
        wrapped = f"```json\n{json.dumps(mock_response)}\n```"

        with patch("services.llm.httpx.post", return_value=_mock_http_response(wrapped)):
            result = call_final_report_model(
                input_context="Dynamic context",
                conversation_messages=[
                    {"role": "assistant", "content": "What is the patient's age?"},
                    {"role": "user", "content": "52 years old."},
                ],
            )

        self.assertEqual(result["mode"], "final_report")
        self.assertEqual(result["who_grade_predicted"], "4")
        self.assertEqual(result["diagnosis_label"], "Glioblastoma")

    def test_call_interview_model_with_invalid_json_raises(self):
        with patch("services.llm.httpx.post", return_value=_mock_http_response("This is not JSON")):
            with self.assertRaises(LLMResponseError):
                call_interview_model(
                    input_context="Dynamic context",
                    conversation_messages=[],
                    max_question_turns=15,
                )


if __name__ == "__main__":
    unittest.main()
