"""Budget guardrails — enforced BEFORE calls fire (MAR-17).

Model-free: pricing is set on the ModelSpec and the provider call is monkeypatched
to a counter, so we can prove no call happens when a cap would be breached.
"""

import asyncio

import pytest

from autofusion.budget import BudgetExceeded, BudgetTracker, estimate_request_cost
from autofusion.config import ModelSpec
from autofusion.providers import CompletionResult
from autofusion.strategies import Fusion, SingleModel


def _paid(name, in_cost=1e-3, out_cost=1e-3):
    return ModelSpec(name=name, model=name, input_cost_per_token=in_cost, output_cost_per_token=out_cost)


def _free(name):
    return ModelSpec(name=name, model=f"ollama/{name}", input_cost_per_token=0.0, output_cost_per_token=0.0)


def _patch_provider(monkeypatch):
    """Replace acomplete with a counter; returns the call-count dict."""
    import autofusion.strategies as strat
    calls = {"n": 0}

    async def fake_acomplete(spec, messages, **kw):
        calls["n"] += 1
        return CompletionResult(spec.name, "ok", cost_usd=0.01, latency_s=0.0,
                                prompt_tokens=1, completion_tokens=1)

    monkeypatch.setattr(strat, "acomplete", fake_acomplete)
    return calls


def test_free_model_estimates_zero():
    assert estimate_request_cost(_free("llama3.2"), [{"role": "user", "content": "hi"}]) == 0.0


def test_over_per_request_cap_blocks_before_any_call(monkeypatch):
    calls = _patch_provider(monkeypatch)
    # Tiny per-request cap; a paid model's projected cost will exceed it.
    budget = BudgetTracker(per_request_usd=1e-9)
    strat = SingleModel(_paid("gpt-x"))
    with pytest.raises(BudgetExceeded):
        asyncio.run(strat.run([{"role": "user", "content": "expensive request here"}], budget=budget))
    assert calls["n"] == 0  # the guard fired BEFORE the provider was called


def test_free_model_never_blocked(monkeypatch):
    calls = _patch_provider(monkeypatch)
    budget = BudgetTracker(per_request_usd=1e-12, total_usd=1e-12)
    strat = SingleModel(_free("llama3.2"))
    result = asyncio.run(strat.run([{"role": "user", "content": "hi"}], budget=budget))
    assert result.ok and calls["n"] == 1


def test_cumulative_total_cap_trips_next_call(monkeypatch):
    _patch_provider(monkeypatch)
    # Generous per-request cap, tight total. First call ok; spend accrues; next trips.
    budget = BudgetTracker(per_request_usd=1.0, total_usd=0.015)
    strat = SingleModel(_paid("gpt-x", in_cost=0.0, out_cost=0.0))
    # Force a known projected cost by stubbing the estimator via spec pricing:
    paid = _paid("gpt-x", in_cost=1e-4, out_cost=1e-4)
    strat = SingleModel(paid)
    # First request records ~0.01 actual; do it twice and expect the 2nd-or-3rd to trip.
    tripped = False
    for _ in range(5):
        try:
            asyncio.run(strat.run([{"role": "user", "content": "x" * 50}], budget=budget))
        except BudgetExceeded:
            tripped = True
            break
    assert tripped


def test_status_line_renders():
    line = BudgetTracker(per_request_usd=0.5, total_usd=None).status_line()
    assert "per-request cap: $0.5000" in line and "total cap: unlimited" in line
