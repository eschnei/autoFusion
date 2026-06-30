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

# Starter config written by `autofusion init` — local Ollama $0 profile by default.
STARTER_CONFIG = """\
# autoFusion configuration. API keys are read from the environment / .env,
# NEVER stored here. Per-token cost 0 = free/local (skips budget checks).

[[models]]
name = "llama3.2"
model = "ollama/llama3.2"
api_base = "http://localhost:11434"
input_cost_per_token = 0.0
output_cost_per_token = 0.0

[[models]]
name = "qwen2.5"
model = "ollama/qwen2.5:3b"
api_base = "http://localhost:11434"
input_cost_per_token = 0.0
output_cost_per_token = 0.0

# Example hosted model (uncomment + add OPENAI_API_KEY to .env to use):
# [[models]]
# name = "gpt-4o-mini"
# model = "gpt-4o-mini"

[fusion]
proposers = ["llama3.2", "qwen2.5"]
aggregator = "llama3.2"
layers = 1

[router]
# Picks ONE model per request: first matching rule wins, else `default`.
default = "llama3.2"
[[router.rules]]
match = "code|function|algorithm|def |class |array|sort|implement"
model = "qwen2.5"

[cascade]
# Cheapest-first; a critic scores each answer and escalates below `threshold`.
tiers = ["llama3.2", "qwen2.5"]
critic = "llama3.2"
threshold = 0.7

[budget]
# Hard caps enforced before calls fire. null = unlimited.
per_request_usd = 0.50
total_usd = 10.0
"""


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
class RouterConfig:
    """Heuristic router: ordered (regex, model-name) rules + a default."""

    default: str | None = None
    rules: list[tuple[str, str]] = field(default_factory=list)  # (pattern, model name)


@dataclass
class CascadeConfig:
    """Cost cascade: try `tiers` cheapest-first, a `critic` gates escalation."""

    tiers: list[str] = field(default_factory=list)  # strategy names, cheapest -> priciest
    critic: str | None = None
    threshold: float = 0.7


@dataclass
class Config:
    models: dict[str, ModelSpec]
    fusion: FusionConfig
    budget: BudgetConfig
    router: RouterConfig = field(default_factory=RouterConfig)
    cascade: CascadeConfig = field(default_factory=CascadeConfig)
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
    r = raw.get("router", {})
    router = RouterConfig(
        default=r.get("default"),
        rules=[(rule["match"], rule["model"]) for rule in r.get("rules", [])],
    )
    c = raw.get("cascade", {})
    cascade = CascadeConfig(
        tiers=list(c.get("tiers", [])),
        critic=c.get("critic"),
        threshold=float(c.get("threshold", 0.7)),
    )
    return Config(
        models=models, fusion=fusion, budget=budget,
        router=router, cascade=cascade, path=path,
    )
