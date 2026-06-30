"""Capability/cost registry (MAR-26)."""

from autofusion.config import BudgetConfig, Config, FusionConfig, ModelSpec
from autofusion.registry import build_registry, models_for_capability, profile_for


def _spec(name, model=None, in_c=None, out_c=None, caps=()):
    return ModelSpec(name=name, model=model or name, input_cost_per_token=in_c,
                     output_cost_per_token=out_c, capabilities=caps)


def test_config_cost_override_beats_litellm():
    p = profile_for(_spec("gpt-4o", in_c=1.0e-6, out_c=2.0e-6, caps=("code",)))
    assert p.input_cost_per_token == 1.0e-6
    assert p.output_cost_per_mtok == 2.0      # 2e-6 * 1e6
    assert "code" in p.capabilities


def test_local_model_is_free_and_flagged():
    p = profile_for(_spec("llama3.2", model="ollama/llama3.2", in_c=0.0, out_c=0.0,
                          caps=("general",)))
    assert p.is_local and p.input_cost_per_token == 0.0


def _config():
    return Config(
        models={
            "cheap": _spec("cheap", model="ollama/cheap", in_c=0.0, out_c=0.0, caps=("code", "general")),
            "mid": _spec("mid", in_c=1e-6, out_c=3e-6, caps=("code",)),
            "pricey": _spec("pricey", in_c=5e-6, out_c=15e-6, caps=("code", "reasoning")),
        },
        fusion=FusionConfig(),
        budget=BudgetConfig(),
    )


def test_build_registry_and_capability_lookup():
    reg = build_registry(_config())
    assert set(reg) == {"cheap", "mid", "pricey"}
    coders = models_for_capability(reg, "code")
    assert [p.name for p in coders] == ["cheap", "mid", "pricey"]   # sorted by output cost
    assert [p.name for p in models_for_capability(reg, "reasoning")] == ["pricey"]
    assert models_for_capability(reg, "vision") == []
