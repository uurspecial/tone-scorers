"""
title: Chinese Tone Scorers
author: tone-scorers
version: 1.0.0
description: 用四個 Workbook3 分類器評分指定文字，取得語氣及各類別機率。
requirements: httpx
"""
from pydantic import BaseModel, Field
import httpx


class Tools:
    class Valves(BaseModel):
        SCORER_URL: str = Field(default="http://host.docker.internal:8001", description="Scorer server URL without /v1")
        SCORER_API_KEY: str = Field(default="", description="Scorer API key", json_schema_extra={"input": {"type": "password"}})

    def __init__(self):
        self.valves = self.Valves()

    async def score_tone(self, text: str, model: str = "roberta-base") -> dict:
        """辨識指定文字的語氣。模型可選 roberta-base、roberta-large、llama-8b-lora、llama-8b-4bit-lora。

        :param text: 要分類的原始文字；可傳入使用者文字或模型生成的回答。
        :param model: 分類器 ID。
        :return: 預測語氣、四個類別的機率與 logits。
        """
        models = {"roberta-base", "roberta-large", "llama-8b-lora", "llama-8b-4bit-lora"}
        if model not in models:
            return {"error": "Unknown scorer model", "models": sorted(models)}
        if not self.valves.SCORER_API_KEY:
            return {"error": "Configure SCORER_API_KEY in the tool's Valves"}
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                response = await client.post(
                    self.valves.SCORER_URL.rstrip("/") + "/v1/score",
                    headers={"Authorization": "Bearer " + self.valves.SCORER_API_KEY},
                    json={"model": model, "text": text},
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError:
            return {"error": "Scorer request failed; check the URL, API key, server logs and GPU memory"}
