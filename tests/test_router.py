"""Heuristic router (MAR-23), model-free."""

import asyncio
import re

from autofusion.config import BudgetConfig, Config, FusionConfig, ModelSpec, RouterConfig
from autofusion.providers import CompletionResult
from autofusion.strategies import Router, resolve_strategy


def _spec(name):
    return ModelSpec(name=name, model=f"ollama/{name}", input_cost_per_token=0.0,
                     output_cost_per_token=0.0)


def _router():
    return Router(
        default=_spec("llama3.2"),
        rules=[(re.compile("code|def |algorithm", re.I), _spec("qwen2.5-coder"))],
    )


def test_selects_coder_for_code_prompt():
    r = _router()
    assert r.select([{"role": "user", "content": "Write code to sort an array"}]).name == "qwen2.5-coder"


def test_selects_default_otherwise():
    r = _router()
    assert r.select([{"role": "user", "content": "Tell me a joke"}]).name == "llama3.2"


def test_run_makes_exactly_one_call_to_selected(monkeypatch):
    import autofusion.strategies as strat
    calls = []

    async def fake_acomplete(spec, messages, **kw):
        calls.append(spec.name)
        return CompletionResult(spec.name, "ok", 0.0, 0.0, 1, 1)

    monkeypatch.setattr(strat, "acomplete", fake_acomplete)
    result = asyncio.run(_router().run([{"role": "user", "content": "write code to sort"}]))
    assert calls == ["qwen2.5-coder"]          # exactly one call, to the routed model
    assert result.model == "qwen2.5-coder"


def test_resolve_strategy_builds_router():
    cfg = Config(
        models={"llama3.2": _spec("llama3.2"), "qwen2.5-coder": _spec("qwen2.5-coder")},
        fusion=FusionConfig(),
        budget=BudgetConfig(),
        router=RouterConfig(default="llama3.2", rules=[("code|def ", "qwen2.5-coder")]),
    )
    strat = resolve_strategy(cfg, "route")
    assert isinstance(strat, Router)
    assert strat.default.name == "llama3.2" and len(strat.rules) == 1
