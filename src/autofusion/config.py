"""Configuration loading (Phase 0).

Reads a TOML config file for the model registry + settings, and provider API
keys from the environment / .env. Keys are never read from the TOML.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_CONFIG_NAME = "autofusion.toml"


@dataclass(frozen=True)
class ModelSpec:
    """One entry in the model registry."""

    name: str  # short alias used on the CLI / in fusion config
    model: str  # litellm model id, e.g. "ollama/llama3.2" or "gpt-4o-mini"
    api_base: str | None = None
    input_cost_per_token: float | None = None
    output_cost_per_token: float | None = None
    extra: dict = field(default_factory=dict)  # passthrough litellm params

    @property
    def is_local(self) -> bool:
        return self.model.startswith("ollama/") or self.model.startswith("vllm/")


@dataclass
class FusionConfig:
    proposers: list[str] = field(default_factory=list)
    aggregator: str | None = None
    layers: int = 1


@dataclass
class BudgetConfig:
    per_request_usd: float | None = None
    total_usd: float | None = None


@dataclass
class Config:
    models: dict[str, ModelSpec]
    fusion: FusionConfig
    budget: BudgetConfig
    path: Path | None = None

    def model(self, name: str) -> ModelSpec:
        try:
            return self.models[name]
        except KeyError:
            known = ", ".join(sorted(self.models)) or "(none)"
            raise KeyError(f"unknown model '{name}'. Configured: {known}") from None


def find_config(explicit: str | os.PathLike | None = None) -> Path:
    """Locate the config file: explicit path, else nearest autofusion.toml."""
    if explicit:
        p = Path(explicit).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"config not found: {p}")
        return p
    for directory in (Path.cwd(), *Path.cwd().parents):
        candidate = directory / DEFAULT_CONFIG_NAME
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"no {DEFAULT_CONFIG_NAME} found in {Path.cwd()} or any parent. "
        "Run `autofusion init` or create one."
    )


def load_config(explicit: str | os.PathLike | None = None) -> Config:
    """Load and parse config; also loads .env so provider keys are visible."""
    load_dotenv()  # populate os.environ from a local .env if present
    path = find_config(explicit)
    with path.open("rb") as fh:
        raw = tomllib.load(fh)

    models: dict[str, ModelSpec] = {}
    for entry in raw.get("models", []):
        known = {"name", "model", "api_base", "input_cost_per_token", "output_cost_per_token"}
        spec = ModelSpec(
            name=entry["name"],
            model=entry["model"],
            api_base=entry.get("api_base"),
            input_cost_per_token=entry.get("input_cost_per_token"),
            output_cost_per_token=entry.get("output_cost_per_token"),
            extra={k: v for k, v in entry.items() if k not in known},
        )
        if spec.name in models:
            raise ValueError(f"duplicate model name in config: {spec.name}")
        models[spec.name] = spec

    f = raw.get("fusion", {})
    fusion = FusionConfig(
        proposers=list(f.get("proposers", [])),
        aggregator=f.get("aggregator"),
        layers=int(f.get("layers", 1)),
    )
    b = raw.get("budget", {})
    budget = BudgetConfig(
        per_request_usd=b.get("per_request_usd"),
        total_usd=b.get("total_usd"),
    )
    return Config(models=models, fusion=fusion, budget=budget, path=path)
