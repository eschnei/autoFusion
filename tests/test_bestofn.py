"""Verified best-of-N (MAR-28), model-free."""

import asyncio

import autofusion.strategies as strat
from autofusion.config import (
    BestOfNConfig, BudgetConfig, Config, FusionConfig, ModelSpec, RouterConfig,
)
from autofusion.providers import CompletionResult
from autofusion.strategies import VerifiedBestOfN, resolve_strategy


def _spec(name):
    return ModelSpec(name=name, model=f"ollama/{name}", input_cost_per_token=0.0,
                     output_cost_per_token=0.0)


def _patch_candidates(monkeypatch, texts):
    """acomplete returns the next queued text each call; records nothing else."""
    seq = iter(texts)

    async def fake_acomplete(spec, messages, **kw):
        return CompletionResult(spec.name, next(seq), 0.01, 0.1, 1, 1)

    monkeypatch.setattr(strat, "acomplete", fake_acomplete)


def test_verifier_picks_first_passing_candidate(monkeypatch):
    _patch_candidates(monkeypatch, ["wrong1", "wrong2", "RIGHT", "wrong3"])
    bon = VerifiedBestOfN(models=[_spec("a")], n=4, temperature=0.7)
    verify = lambda text: text == "RIGHT"
    r = asyncio.run(bon.run([{"role": "user", "content": "q"}], verify=verify))
    assert r.text == "RIGHT"
    assert r.n_calls == 4                       # all 4 sampled
    assert abs(r.cost_usd - 0.04) < 1e-9


def test_no_verifier_falls_back_to_critic(monkeypatch):
    _patch_candidates(monkeypatch, ["meh", "best", "ok"])

    async def fake_critic(critic, messages, answer, budget, **kw):
        score = {"meh": 0.1, "best": 0.9, "ok": 0.5}[answer]
        return score, CompletionResult("critic", "", 0.001, 0.0, 1, 1)

    monkeypatch.setattr(strat, "_critic_score", fake_critic)
    bon = VerifiedBestOfN(models=[_spec("a")], n=3, critic=_spec("critic"))
    r = asyncio.run(bon.run([{"role": "user", "content": "q"}]))   # no verify
    assert r.text == "best"
    assert r.n_calls == 6                        # 3 candidates + 3 critic calls


def test_needs_verifier_flag_set():
    assert VerifiedBestOfN(models=[_spec("a")]).needs_verifier is True


def test_no_passing_candidate_does_not_crash(monkeypatch):
    _patch_candidates(monkeypatch, ["x", "y"])
    bon = VerifiedBestOfN(models=[_spec("a")], n=2)   # no critic
    r = asyncio.run(bon.run([{"role": "user", "content": "q"}], verify=lambda t: False))
    assert r.ok and r.text in ("x", "y")          # returns something, honest

def test_resolve_strategy_builds_bestofn():
    cfg = Config(
        models={n: _spec(n) for n in ("a", "b", "critic")},
        fusion=FusionConfig(), budget=BudgetConfig(), router=RouterConfig(),
        bestofn=BestOfNConfig(models=["a", "b"], n=5, critic="critic", temperature=0.5),
    )
    s = resolve_strategy(cfg, "bestofn")
    assert isinstance(s, VerifiedBestOfN)
    assert [m.name for m in s.models] == ["a", "b"] and s.n == 5 and s.critic.name == "critic"
