"""Cost cascade (MAR-25), model-free.

A stub strategy returns a tagged answer per tier; the critic's score is
controlled by monkeypatching _critic_score so we can drive escalation.
"""

import asyncio

import autofusion.strategies as strat
from autofusion.config import (
    BudgetConfig, CascadeConfig, Config, FusionConfig, ModelSpec, RouterConfig,
)
from autofusion.providers import CompletionResult
from autofusion.strategies import Cascade, resolve_strategy


def _spec(name):
    return ModelSpec(name=name, model=f"ollama/{name}", input_cost_per_token=0.0,
                     output_cost_per_token=0.0)


class _Tier:
    """A fake strategy: one call, fixed cost, tags its tier name."""

    def __init__(self, name, cost=0.01):
        self.name = name
        self._cost = cost
        self.calls = 0

    async def run(self, messages, budget=None, **kw):
        self.calls += 1
        return CompletionResult(self.name, f"answer-{self.name}", self._cost, 0.1, 1, 1)


def _patch_critic(monkeypatch, score):
    async def fake_critic(critic, messages, answer, budget, **kw):
        return score, CompletionResult("critic", "", 0.001, 0.0, 1, 1)
    monkeypatch.setattr(strat, "_critic_score", fake_critic)


def test_confident_stops_at_cheap_tier(monkeypatch):
    _patch_critic(monkeypatch, score=0.95)  # >= threshold
    cheap, pricey = _Tier("cheap"), _Tier("pricey")
    c = Cascade(tiers=[cheap, pricey], critic=_spec("critic"), threshold=0.7)
    r = asyncio.run(c.run([{"role": "user", "content": "q"}]))
    assert r.text == "answer-cheap"
    assert cheap.calls == 1 and pricey.calls == 0          # no escalation
    assert r.n_calls == 2                                   # cheap + critic


def test_low_confidence_escalates(monkeypatch):
    _patch_critic(monkeypatch, score=0.2)  # < threshold
    cheap, pricey = _Tier("cheap"), _Tier("pricey")
    c = Cascade(tiers=[cheap, pricey], critic=_spec("critic"), threshold=0.7)
    r = asyncio.run(c.run([{"role": "user", "content": "q"}]))
    assert r.text == "answer-pricey"                        # escalated
    assert cheap.calls == 1 and pricey.calls == 1
    assert abs(r.cost_usd - (0.01 + 0.001 + 0.01)) < 1e-9   # cheap + critic + pricey
    assert r.n_calls == 3


def test_unparseable_critic_fails_safe_to_escalation(monkeypatch):
    _patch_critic(monkeypatch, score=None)  # critic gave no number
    cheap, pricey = _Tier("cheap"), _Tier("pricey")
    c = Cascade(tiers=[cheap, pricey], critic=_spec("critic"), threshold=0.7)
    r = asyncio.run(c.run([{"role": "user", "content": "q"}]))
    assert r.text == "answer-pricey" and pricey.calls == 1  # escalated, not trusted


def test_resolve_strategy_builds_cascade_with_fusion_tier():
    cfg = Config(
        models={n: _spec(n) for n in ("a", "b", "critic")},
        fusion=FusionConfig(proposers=["a", "b"], aggregator="b", layers=1),
        budget=BudgetConfig(),
        router=RouterConfig(),
        cascade=CascadeConfig(tiers=["a", "fusion"], critic="critic", threshold=0.6),
    )
    c = resolve_strategy(cfg, "cascade")
    assert isinstance(c, Cascade)
    assert [t.name for t in c.tiers] == ["a", "fusionMarj"]  # a tier can be fusion
    assert c.critic.name == "critic" and c.threshold == 0.6


def test_cascade_requires_two_tiers_and_critic():
    cfg = Config(models={"a": _spec("a")}, fusion=FusionConfig(), budget=BudgetConfig(),
                 router=RouterConfig(), cascade=CascadeConfig(tiers=["a"], critic=None))
    try:
        resolve_strategy(cfg, "cascade")
        assert False, "should have raised"
    except ValueError:
        pass
