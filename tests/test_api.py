import json
import unittest

from fastapi.testclient import TestClient
from artifacts import read_manifest
from server import create_app


class FakeScorers:
    def __init__(self):
        self.descriptors = {m["id"]: m for m in read_manifest()["models"]}
        self.loaded_model = None
        self.calls = []

    def predict(self, model, text):
        self.calls.append((model, text))
        return {"model": model, "label_id": 4, "label": "酸", "input_tokens": 12,
                "probabilities": {"溫和": 0.05, "中性": 0.05, "不滿": 0.1, "酸": 0.8},
                "logits": [0, 0, 1, 3]}


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.scorers = FakeScorers()
        self.client = TestClient(create_app(self.scorers, api_key="test-key"))
        self.headers = {"Authorization": "Bearer test-key"}

    def test_authentication_and_model_discovery(self):
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.assertEqual(self.client.get("/v1/models").status_code, 401)
        self.assertEqual(self.client.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code, 401)
        response = self.client.get("/v1/models", headers=self.headers)
        self.assertEqual({m["id"] for m in response.json()["data"]}, {
            "roberta-base", "roberta-large", "llama-8b-lora", "llama-8b-4bit-lora"})
        unconfigured = TestClient(create_app(self.scorers, api_key=""))
        self.assertEqual(unconfigured.get("/v1/models").status_code, 503)

    def test_last_user_message_and_nonstream_response(self):
        response = self.client.post("/v1/chat/completions", headers=self.headers, json={
            "model": "roberta-base", "temperature": 0.8,
            "messages": [{"role": "system", "content": "system instructions"},
                         {"role": "user", "content": "old"},
                         {"role": "assistant", "content": "previous reply"},
                         {"role": "user", "content": [{"type": "text", "text": "new text"}]}]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.scorers.calls, [("roberta-base", "new text")])
        self.assertEqual(response.json()["choices"][0]["finish_reason"], "stop")
        self.assertIn("酸", response.json()["choices"][0]["message"]["content"])

    def test_streaming_chunks_reconstruct_same_answer(self):
        body = {"model": "llama-8b-4bit-lora", "messages": [{"role": "user", "content": "測試"}]}
        ordinary = self.client.post("/v1/chat/completions", headers=self.headers, json=body).json()
        response = self.client.post("/v1/chat/completions", headers=self.headers,
                                    json={**body, "stream": True, "stream_options": {"include_usage": True}})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        lines = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
        self.assertEqual(lines[-1], "[DONE]")
        chunks = [json.loads(line) for line in lines[:-1]]
        content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
        self.assertEqual(content, ordinary["choices"][0]["message"]["content"])
        self.assertEqual(chunks[-2]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(chunks[-1]["usage"]["prompt_tokens"], 12)

    def test_rejects_invalid_inputs_before_inference(self):
        for messages in [[], [{"role": "system", "content": "x"}],
                         [{"role": "user", "content": " "}],
                         [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]]:
            response = self.client.post("/v1/chat/completions", headers=self.headers,
                                        json={"model": "roberta-base", "messages": messages})
            self.assertIn(response.status_code, {400, 422})
        self.assertEqual(self.scorers.calls, [])
        response = self.client.post("/v1/score", headers=self.headers, json={"model": "unknown", "text": "x"})
        self.assertEqual(response.status_code, 404)
        self.assertEqual(self.client.post("/v1/score", headers=self.headers,
                         json={"model": "roberta-base", "text": "x" * 30001}).status_code, 422)

    def test_score_endpoint_returns_structured_result(self):
        response = self.client.post("/v1/score", headers=self.headers,
                                    json={"model": "roberta-large", "text": "原始文字"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["label_id"], 4)
        self.assertAlmostEqual(sum(response.json()["probabilities"].values()), 1)


if __name__ == "__main__":
    unittest.main()
