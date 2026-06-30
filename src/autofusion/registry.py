"""Capability + cost registry (foundation for capability-aware fusion).

Per-model profile = what it's best at (config `capabilities`) + price (config
override, else LiteLLM's `model_cost`) + context window. Lets fusion/route/
cascade pick the right diverse proposers by capability and cost instead of
guessing — the prerequisite for "fuse cheaper strong models to match Opus".
"""

from __future__ import annotations

from dataclasses import dataclass

import litellm

from .budget import _per_token_costs
from .config import Config, ModelSpec


@dataclass
class CapabilityProfile:
    name: str
    model: str
    is_local: bool
    input_cost_per_token: float
    output_cost_per_token: float
    context_window: int | None
    capabilities: tuple[str, ...]

    @property
    def input_cost_per_mtok(self) -> float:
        return self.input_cost_per_token * 1_000_000

    @property
    def output_cost_per_mtok(self) -> float:
        return self.output_cost_per_token * 1_000_000


def profile_for(spec: ModelSpec) -> CapabilityProfile:
    in_cost, out_cost = _per_token_costs(spec)  # config override beats LiteLLM
    info = litellm.model_cost.get(spec.model) or {}
    context = info.get("max_input_tokens") or info.get("max_tokens")
    return CapabilityProfile(
        name=spec.name, model=spec.model, is_local=spec.is_local,
        input_cost_per_token=in_cost, output_cost_per_token=out_cost,
        context_window=context, capabilities=spec.capabilities,
    )


def build_registry(config: Config) -> dict[str, CapabilityProfile]:
    return {name: profile_for(spec) for name, spec in config.models.items()}


def models_for_capability(
    registry: dict[str, CapabilityProfile], capability: str
) -> list[CapabilityProfile]:
    """Models tagged with `capability`, cheapest first (by output cost)."""
    matches = [p for p in registry.values() if capability in p.capabilities]
    return sorted(matches, key=lambda p: p.output_cost_per_token)
