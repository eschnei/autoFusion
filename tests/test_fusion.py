"""Model-free checks of fusion plumbing.

A bug in how candidates are assembled into the aggregator prompt would silently
degrade every fused answer while the eval still ran green — so verify the
construction deterministically, no model in the loop.
"""

import asyncio

from autofusion.config import ModelSpec
from autofusion.strategies import Fusion, build_aggregate_messages


def _spec(name):
    return ModelSpec(name=name, model=f"ollama/{name}", input_cost_per_token=0.0,
                     output_cost_per_token=0.0)


def test_aggregate_messages_preserve_original_and_carry_all_candidates():
    original = [
        {"role": "system", "content": "Return only a code block."},
        {"role": "user", "content": "Write is_even(n)."},
    ]
    msgs = build_aggregate_messages(original, ["draft A", "draft B"])
    # Original turns untouched and still first (format instruction preserved).
    assert msgs[:2] == original
    # Exactly one synthesis turn appended, as a user turn.
    assert len(msgs) == 3
    assert msgs[-1]["role"] == "user"
    # Every candidate is present in the synthesis turn.
    assert "draft A" in msgs[-1]["content"]
    assert "draft B" in msgs[-1]["content"]


def test_fusion_accumulates_cost_and_calls(monkeypatch):
    """Fusion result must sum cost + count every proposer and aggregator call."""
    import autofusion.strategies as strat

    calls = {"n": 0}

    async def fake_acomplete(spec, messages, **kw):
        from autofusion.providers import CompletionResult
        calls["n"] += 1
        return CompletionResult(spec.name, f"answer from {spec.name}", cost_usd=0.01,
                                latency_s=0.1, prompt_tokens=1, completion_tokens=1)

    monkeypatch.setattr(strat, "acomplete", fake_acomplete)

    fusion = Fusion(proposers=[_spec("a"), _spec("b")], aggregator=_spec("agg"), layers=1)
    result = asyncio.run(fusion.run([{"role": "user", "content": "hi"}]))

    assert calls["n"] == 3  # 2 proposers + 1 aggregator
    assert result.n_calls == 3
    assert abs(result.cost_usd - 0.03) < 1e-9
    assert result.text == "answer from agg"
