"""OpenAI-compatible transport for four fixed Chinese tone classifiers."""
from __future__ import annotations

import json
import logging
import os
import secrets
import time
import uuid
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from runtime import ScorerRegistry

logger = logging.getLogger(__name__)


class ScoreRequest(BaseModel):
    model: str
    text: str = Field(min_length=1, max_length=30000)


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]] | None = None


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    stream_options: dict[str, Any] | None = None


def last_user_text(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role != "user":
            continue
        content = message.content
        if isinstance(content, list):
            if any(part.get("type") != "text" or not isinstance(part.get("text"), str) for part in content):
                raise HTTPException(400, "Scorers accept text input only")
            content = "\n".join(part["text"] for part in content)
        if not isinstance(content, str) or not content.strip():
            raise HTTPException(400, "The last user message must contain text")
        if len(content) > 30000:
            raise HTTPException(400, "Input exceeds 30000 characters")
        return content
    raise HTTPException(400, "A user message is required")


def render_result(result: dict) -> str:
    lines = [f"預測語氣：**{result['label']}**", "", "| 語氣 | 分類機率 |", "| --- | ---: |"]
    lines.extend(f"| {label} | {value:.2%} |" for label, value in result["probabilities"].items())
    return "\n".join(lines)


def create_app(predictor=None, api_key: str | None = None) -> FastAPI:
    registry = predictor or ScorerRegistry()
    expected_key = api_key if api_key is not None else os.environ.get("SCORER_API_KEY", "")
    bearer = HTTPBearer(auto_error=False)
    app = FastAPI(title="Chinese tone scorers", version="1.0.0")
    app.state.registry = registry

    async def authorize(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
        if not expected_key or expected_key == "replace-with-a-random-api-key":
            raise HTTPException(503, "Set SCORER_API_KEY before using this service")
        if credentials is None or credentials.scheme.lower() != "bearer" or not secrets.compare_digest(
            credentials.credentials.encode(), expected_key.encode()
        ):
            raise HTTPException(401, "Invalid API key", headers={"WWW-Authenticate": "Bearer"})

    @app.exception_handler(HTTPException)
    async def http_error(_request, error):
        return JSONResponse(
            status_code=error.status_code,
            content={"error": {"message": str(error.detail), "type": "invalid_request_error", "code": error.status_code}},
            headers=error.headers,
        )

    async def score(model: str, text: str) -> dict:
        if model not in registry.descriptors:
            raise HTTPException(404, "Unknown scorer model")
        if not text.strip():
            raise HTTPException(400, "Input text is empty")
        try:
            return await run_in_threadpool(registry.predict, model, text)
        except Exception:
            logger.exception("Scorer inference failed for %s", model)
            raise HTTPException(503, "Model loading or inference failed; check server logs") from None

    @app.get("/health")
    async def health():
        return {"status": "ok", "loaded_model": registry.loaded_model}

    @app.get("/v1/models", dependencies=[Depends(authorize)])
    async def models():
        return {"object": "list", "data": [
            {"id": name, "object": "model", "created": 0, "owned_by": "tone-scorers"}
            for name in registry.descriptors
        ]}

    @app.post("/v1/score", dependencies=[Depends(authorize)])
    async def classify(body: ScoreRequest):
        return await score(body.model, body.text)

    @app.post("/v1/chat/completions", dependencies=[Depends(authorize)])
    async def chat(body: ChatRequest):
        result = await score(body.model, last_user_text(body.messages))
        content = render_result(result)
        completion_id = "chatcmpl-" + uuid.uuid4().hex
        created = int(time.time())
        usage = {"prompt_tokens": result["input_tokens"], "completion_tokens": 0, "total_tokens": result["input_tokens"]}
        common = {"id": completion_id, "created": created, "model": body.model}
        if not body.stream:
            return {**common, "object": "chat.completion", "choices": [
                {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
            ], "usage": usage}

        async def events():
            for delta, finish in [({"role": "assistant"}, None), ({"content": content}, None), ({}, "stop")]:
                chunk = {**common, "object": "chat.completion.chunk", "choices": [
                    {"index": 0, "delta": delta, "finish_reason": finish}
                ]}
                yield "data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n"
            if body.stream_options and body.stream_options.get("include_usage"):
                yield "data: " + json.dumps({**common, "object": "chat.completion.chunk", "choices": [], "usage": usage}) + "\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


app = create_app()
