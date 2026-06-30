"""OpenAI-compatible HTTP endpoint (Phase 4, MAR-19).

A thin FastAPI server exposing POST /v1/chat/completions so any OpenAI client
or IDE can point at the harness with a one-line base-URL change. The `model`
field selects a strategy: a configured model name -> single model; "fusion" ->
the [fusion] config. Budget caps (MAR-17) are enforced here too.

Provider normalization stays in LiteLLM; this server only orchestrates the
fusion strategy and reshapes the result into the OpenAI schema.
"""

from __future__ import annotations

import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .budget import BudgetExceeded, BudgetTracker
from .config import Config
from .providers import CompletionResult
from .strategies import resolve_strategy


def _openai_body(model: str, result: CompletionResult) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.prompt_tokens + result.completion_tokens,
        },
        # Non-standard extension: fusion's true cost + call count.
        "x_autofusion": {"cost_usd": result.cost_usd, "n_calls": result.n_calls},
    }


def create_app(config: Config) -> FastAPI:
    app = FastAPI(title="autoFusion", version="0.0.1")
    # Cumulative budget across the server's lifetime.
    app.state.budget = BudgetTracker.from_config(config.budget)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "models": sorted(config.models)}

    @app.post("/v1/chat/completions")
    async def chat_completions(body: dict) -> JSONResponse:
        model = body.get("model")
        messages = body.get("messages")
        if not model or not messages:
            raise HTTPException(status_code=400, detail="`model` and `messages` are required")
        try:
            strategy = resolve_strategy(config, model)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

        kw = {k: body[k] for k in ("temperature", "max_tokens") if k in body}
        try:
            result = await strategy.run(messages, budget=app.state.budget, **kw)
        except BudgetExceeded as exc:
            # 402 Payment Required — the cap fired before spending.
            raise HTTPException(status_code=402, detail=f"budget cap: {exc}") from None
        if not result.ok:
            raise HTTPException(status_code=502, detail=result.error)
        return JSONResponse(_openai_body(model, result))

    return app
