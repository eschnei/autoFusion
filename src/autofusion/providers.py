"""Provider layer (Phase 0).

A single `complete()` / `acomplete()` wrapper over LiteLLM so every model —
hosted or local Ollama — is called through one interface and returns a
normalized result carrying text, cost, latency, and token counts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import litellm

from .config import ModelSpec

# Don't explode on provider-specific params a given model doesn't accept.
litellm.drop_params = True

Message = dict[str, str]


@dataclass
class CompletionResult:
    model: str  # registry alias
    text: str
    cost_usd: float
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _litellm_kwargs(spec: ModelSpec, messages: list[Message], **overrides) -> dict:
    kwargs: dict = {"model": spec.model, "messages": messages}
    if spec.api_base:
        kwargs["api_base"] = spec.api_base
    # Register per-model cost so LiteLLM's cost tracking covers local/custom models.
    if spec.input_cost_per_token is not None or spec.output_cost_per_token is not None:
        kwargs["input_cost_per_token"] = spec.input_cost_per_token or 0.0
        kwargs["output_cost_per_token"] = spec.output_cost_per_token or 0.0
    kwargs.update(spec.extra)
    kwargs.update(overrides)
    return kwargs


def _extract(spec: ModelSpec, response, latency_s: float) -> CompletionResult:
    text = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    cost = (response._hidden_params or {}).get("response_cost")
    if cost is None:
        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:
            cost = 0.0
    return CompletionResult(
        model=spec.name,
        text=text,
        cost_usd=float(cost or 0.0),
        latency_s=latency_s,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def complete(spec: ModelSpec, messages: list[Message], **overrides) -> CompletionResult:
    """Synchronous single completion. Errors are captured, not raised."""
    start = time.perf_counter()
    try:
        resp = litellm.completion(**_litellm_kwargs(spec, messages, **overrides))
        return _extract(spec, resp, time.perf_counter() - start)
    except Exception as exc:  # noqa: BLE001 — eval runner must stay alive
        return CompletionResult(spec.name, "", 0.0, time.perf_counter() - start, 0, 0, error=str(exc))


async def acomplete(spec: ModelSpec, messages: list[Message], **overrides) -> CompletionResult:
    """Async single completion — use with asyncio.gather for parallel fan-out."""
    start = time.perf_counter()
    try:
        resp = await litellm.acompletion(**_litellm_kwargs(spec, messages, **overrides))
        return _extract(spec, resp, time.perf_counter() - start)
    except Exception as exc:  # noqa: BLE001
        return CompletionResult(spec.name, "", 0.0, time.perf_counter() - start, 0, 0, error=str(exc))
