"""OpenAI-compatible endpoint (MAR-19), model-free via TestClient."""

from fastapi.testclient import TestClient

import autofusion.server as server
from autofusion.budget import BudgetExceeded
from autofusion.config import BudgetConfig, Config, FusionConfig, ModelSpec
from autofusion.providers import CompletionResult


def _config():
    spec = ModelSpec(name="llama3.2", model="ollama/llama3.2",
                     input_cost_per_token=0.0, output_cost_per_token=0.0)
    return Config(
        models={"llama3.2": spec},
        fusion=FusionConfig(proposers=["llama3.2"], aggregator="llama3.2", layers=1),
        budget=BudgetConfig(),
    )


class _Stub:
    name = "fusion"

    def __init__(self, raises=None):
        self._raises = raises

    async def run(self, messages, budget=None, **kw):
        if self._raises:
            raise self._raises
        return CompletionResult("fusion", "hello from fusion", cost_usd=0.0042,
                                latency_s=1.2, prompt_tokens=10, completion_tokens=5, n_calls=3)


def test_chat_completions_returns_openai_shape(monkeypatch):
    monkeypatch.setattr(server, "resolve_strategy", lambda cfg, name: _Stub())
    client = TestClient(server.create_app(_config()))
    r = client.post("/v1/chat/completions",
                    json={"model": "fusion", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    b = r.json()
    assert b["object"] == "chat.completion"
    assert b["choices"][0]["message"]["content"] == "hello from fusion"
    assert b["usage"]["total_tokens"] == 15
    assert b["x_autofusion"] == {"cost_usd": 0.0042, "n_calls": 3}


def test_over_budget_returns_402(monkeypatch):
    monkeypatch.setattr(server, "resolve_strategy",
                        lambda cfg, name: _Stub(raises=BudgetExceeded("over cap")))
    client = TestClient(server.create_app(_config()))
    r = client.post("/v1/chat/completions",
                    json={"model": "fusion", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 402
    assert "budget cap" in r.json()["detail"]


def test_missing_fields_400(monkeypatch):
    monkeypatch.setattr(server, "resolve_strategy", lambda cfg, name: _Stub())
    client = TestClient(server.create_app(_config()))
    assert client.post("/v1/chat/completions", json={"model": "fusion"}).status_code == 400


def test_unknown_model_400():
    client = TestClient(server.create_app(_config()))
    r = client.post("/v1/chat/completions",
                    json={"model": "nope", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 400
