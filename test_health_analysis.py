import unittest
import json
import base64
from unittest.mock import patch

from fastapi.testclient import TestClient

from main import app, _call_gemini


class HealthAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.payload = {
            "pet_name": "小花",
            "score": 80,
            "records": [{
                "record_date": "2026-11-05",
                "category": "diet",
                "observations": ["食慾下降", "有嘔吐"],
                "warnings": ["嘔吐"],
            }],
        }

    def test_mock_gemini_response_reaches_frontend_format(self):
        response_data = {"candidates": [{"content": {"parts": [{"text":
            '{"analysis":"最近食慾下降且有嘔吐紀錄。",'
            '"advice":"持續嘔吐時請聯絡獸醫。"}'
        }]}}]}
        with patch("main._call_gemini", return_value=response_data) as gemini:
            response = self.client.post("/analyze-health", json=self.payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["analysis"], "最近食慾下降且有嘔吐紀錄。")
        prompt = gemini.call_args.args[0]
        self.assertIn("小花", prompt)
        self.assertIn("嘔吐", prompt)
        self.assertIn("80", prompt)

    def test_missing_records_are_rejected_before_gemini(self):
        with patch("main._call_gemini") as gemini:
            response = self.client.post("/analyze-health", json={
                **self.payload, "records": [],
            })
        self.assertEqual(response.status_code, 422)
        gemini.assert_not_called()

    def test_gemini_failure_is_explained(self):
        with patch("main._call_gemini", side_effect=RuntimeError("未設定 GEMINI_API_KEY")):
            response = self.client.post("/analyze-health", json=self.payload)
        self.assertEqual(response.status_code, 502)
        self.assertIn("GEMINI_API_KEY", response.json()["detail"])

    def test_empty_response_reports_token_limit(self):
        response_data = {"candidates": [{"finishReason": "MAX_TOKENS"}]}
        with patch("main._call_gemini", return_value=response_data):
            response = self.client.post("/analyze-health", json=self.payload)
        self.assertEqual(response.status_code, 502)
        self.assertIn("maxOutputTokens", response.json()["detail"])

    def test_text_parts_are_combined(self):
        response_data = {"candidates": [{"content": {"parts": [
            {"text": '{"analysis":"觀察",'},
            {"text": '"advice":"建議"}'},
        ]}}]}
        with patch("main._call_gemini", return_value=response_data):
            response = self.client.post("/analyze-health", json=self.payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"analysis": "觀察", "advice": "建議"})

    def test_request_allows_room_for_thinking_and_answer(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def read(self):
                return b'{}'

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "gemini-3.6-flash"}):
            with patch("main.urlopen", return_value=FakeResponse()) as send:
                _call_gemini("test prompt")
        payload = json.loads(send.call_args.args[0].data)
        config = payload["generationConfig"]
        self.assertEqual(config["maxOutputTokens"], 2048)
        self.assertEqual(config["thinkingConfig"]["thinkingLevel"], "minimal")

    def test_medical_note_is_included_as_context_without_changing_score(self):
        note = "醫師建議持續觀察飲水與精神狀況。" * 12
        payload = {**self.payload, "records": [
            *self.payload["records"],
            {"record_date": "2026-11-05", "category": "medical",
             "observations": ["醫師備註：" + note], "warnings": []},
        ]}
        answer = {"candidates": [{"content": {"parts": [{"text":
            '{"analysis":"有醫療紀錄。","advice":"依醫師說明觀察。"}'
        }]}}]}
        with patch("main._call_gemini", return_value=answer) as gemini:
            response = self.client.post("/analyze-health", json=payload)
        self.assertEqual(response.status_code, 200)
        prompt = gemini.call_args.args[0]
        self.assertIn("醫師備註", prompt)
        self.assertIn(note, prompt)
        self.assertIn("80", prompt)

    def test_medical_report_image_is_forwarded_to_gemini(self):
        image = b"\xff\xd8\xff" + b"sample-report"
        payload = {
            **self.payload,
            "report_image_base64": base64.b64encode(image).decode("ascii"),
            "report_image_mime": "image/jpeg",
            "records": [{
                "record_date": "2026-11-05",
                "category": "medical",
                "observations": ["醫師報告全文：" + "觀察飲水。" * 100],
                "warnings": [],
            }],
        }
        answer = {"candidates": [{"content": {"parts": [
            {"text": '{"analysis":"整理報告","advice":"詢問醫師"}'}
        ]}}]}
        with patch("main._call_gemini", return_value=answer) as gemini:
            response = self.client.post("/analyze-health", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(gemini.call_args.args[1], image)
        self.assertEqual(gemini.call_args.args[2], "image/jpeg")

    def test_fake_report_image_is_rejected(self):
        payload = {
            **self.payload,
            "report_image_base64": base64.b64encode(b"not a jpeg").decode(),
            "report_image_mime": "image/jpeg",
        }
        with patch("main._call_gemini") as gemini:
            response = self.client.post("/analyze-health", json=payload)
        self.assertEqual(response.status_code, 422)
        gemini.assert_not_called()

    def test_claim_summary_does_not_invent_eligibility(self):
        response = self.client.post('/insurance/pre-check', json={
            'pet_id': 'pet-1',
            'claim_items': [{
                'record_id': 'visit-1', 'visit_date': '2026-11-15',
                'diagnosis': '看診', 'cost': 0, 'receipt_url': '',
            }],
        })
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result['selected_items_count'], 1)
        self.assertNotIn('eligible_items_count', result)
        self.assertIn('不是理賠資格判定', result['review_note'])


if __name__ == "__main__":
    unittest.main()
