"""Recipe optimizer (Phase 8).

For a "job" (a benchmark, for now), evaluate candidate recipes over the
**available** model pool and report the quality×cost **Pareto frontier** plus a
recommended recipe. This is the learning loop's v1: the eval harness is the
fitness function; later iterations get smarter search + per-job caching.

"Available" = callable right now = local Ollama (always) + any hosted model
whose API key is present. The pool — and therefore the frontier — grows as keys
are added.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .config import Config
from .eval.results import ModelRunResult

# provider -> env var holding its key (mirror of the CLI's table).
_KEY_ENV = {
    "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY",
}


def model_available(spec) -> bool:
    """Local models are always callable; hosted ones need their key present."""
    if spec.is_local:
        return True
    provider = spec.model.split("/")[0] if "/" in spec.model else "openai"
    return bool(os.environ.get(_KEY_ENV.get(provider, f"{provider.upper()}_API_KEY")))


def available_model_names(config: Config) -> list[str]:
    return [name for name, spec in config.models.items() if model_available(spec)]


def candidate_recipes(config: Config, available: list[str]) -> list[str]:
    """Recipe names to evaluate, drawn only from the available pool:
    each available model + any configured composite whose models are all
    available + an auto-fusion of the pool."""
    recipes = list(available)
    avail = set(available)

    f = config.fusion
    if f.proposers and f.aggregator and avail.issuperset(f.proposers) and f.aggregator in avail:
        recipes.append("fusion")
    r = config.router
    if r.default in avail and all(m in avail for _, m in r.rules):
        recipes.append("route")
    c = config.cascade
    if len(c.tiers) >= 2 and c.critic in avail and all(
        t in avail or t in ("fusion", "route") for t in c.tiers
    ):
        recipes.append("cascade")
    return recipes


@dataclass
class RecipeOutcome:
    recipe: str
    pass_at_1: float
    avg_cost_usd: float
    avg_calls: float
    avg_latency_s: float
    on_frontier: bool = False

    @classmethod
    def from_run(cls, r: ModelRunResult) -> "RecipeOutcome":
        n = r.n_tasks or 1
        return cls(
            recipe=r.model, pass_at_1=r.pass_at_1,
            avg_cost_usd=r.total_cost_usd / n, avg_calls=r.avg_calls,
            avg_latency_s=r.avg_latency_s,
        )


def mark_pareto(outcomes: list[RecipeOutcome]) -> list[RecipeOutcome]:
    """Flag recipes on the quality×cost frontier. A recipe is dominated if
    another has quality >= it AND cost <= it, with at least one strictly better."""
    for a in outcomes:
        dominated = any(
            b is not a
            and b.pass_at_1 >= a.pass_at_1
            and b.avg_cost_usd <= a.avg_cost_usd
            and (b.pass_at_1 > a.pass_at_1 or b.avg_cost_usd < a.avg_cost_usd)
            for b in outcomes
        )
        a.on_frontier = not dominated
    return outcomes


def recommend(outcomes: list[RecipeOutcome]) -> dict[str, RecipeOutcome | None]:
    """Best-quality recipe and the cheapest recipe on the frontier."""
    frontier = [o for o in outcomes if o.on_frontier]
    best_quality = max(outcomes, key=lambda o: (o.pass_at_1, -o.avg_cost_usd), default=None)
    cheapest_frontier = min(frontier, key=lambda o: o.avg_cost_usd, default=None)
    return {"best_quality": best_quality, "cheapest_on_frontier": cheapest_frontier}


def render(outcomes: list[RecipeOutcome], available: list[str]) -> str:
    ranked = sorted(outcomes, key=lambda o: o.pass_at_1, reverse=True)
    lines = [
        f"available models: {', '.join(available) or '(none)'}",
        "",
        f"{'recipe':<18}{'pass@1':>9}{'$/task':>11}{'calls/q':>9}{'lat(s)':>9}{'frontier':>10}",
        "-" * 66,
    ]
    for o in ranked:
        lines.append(
            f"{o.recipe:<18}{o.pass_at_1:>8.1%}{o.avg_cost_usd:>11.5f}"
            f"{o.avg_calls:>9.1f}{o.avg_latency_s:>9.2f}{'  ★' if o.on_frontier else '':>10}"
        )
    rec = recommend(outcomes)
    lines += ["", "recommended:"]
    if rec["best_quality"]:
        b = rec["best_quality"]
        lines.append(f"  highest quality : {b.recipe} ({b.pass_at_1:.1%}, ${b.avg_cost_usd:.5f}/task)")
    if rec["cheapest_on_frontier"]:
        c = rec["cheapest_on_frontier"]
        lines.append(f"  best value      : {c.recipe} ({c.pass_at_1:.1%}, ${c.avg_cost_usd:.5f}/task)")
    return "\n".join(lines)
