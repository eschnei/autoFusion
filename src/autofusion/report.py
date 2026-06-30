"""Cross-task comparison report (Phase 10).

The scoreboard: autoFusion recipes (fuse/route/cascade/bestofn) vs. every
available single model, across several task types (code + math), with quality
and cost. Availability-gated — frontier models join as rows once their keys are
present, turning this into the real "us vs. the big models" comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

from .budget import BudgetTracker
from .config import Config
from .eval.runner import run_baseline
from .optimizer import RecipeOutcome, available_model_names, candidate_recipes


@dataclass
class ReportRow:
    recipe: str
    is_model: bool  # True = a single model ("them"); False = an autoFusion recipe ("us")
    by_benchmark: dict[str, RecipeOutcome]

    def avg_pass(self, benchmarks: list[str]) -> float:
        vals = [self.by_benchmark[b].pass_at_1 for b in benchmarks if b in self.by_benchmark]
        return sum(vals) / len(vals) if vals else 0.0

    def avg_cost(self, benchmarks: list[str]) -> float:
        vals = [self.by_benchmark[b].avg_cost_usd for b in benchmarks if b in self.by_benchmark]
        return sum(vals) / len(vals) if vals else 0.0


async def run_report(
    config: Config, benchmark_names: list[str], limit: int | None = None, concurrency: int = 4
) -> tuple[list[ReportRow], list[str]]:
    available = available_model_names(config)
    recipes = candidate_recipes(config, available)
    model_names = set(config.models)
    budget = BudgetTracker.from_config(config.budget)  # ONE cap across all benchmarks

    by_recipe: dict[str, dict[str, RecipeOutcome]] = {r: {} for r in recipes}
    for bname in benchmark_names:
        results = await run_baseline(
            config, recipes, bname, limit=limit, concurrency=concurrency, budget=budget
        )
        for res in results:
            by_recipe[res.model][bname] = RecipeOutcome.from_run(res)

    rows = [
        ReportRow(recipe=r, is_model=(r in model_names), by_benchmark=by_recipe[r])
        for r in recipes
    ]
    return rows, available


def render_report(rows: list[ReportRow], benchmarks: list[str], available: list[str]) -> str:
    ranked = sorted(rows, key=lambda r: r.avg_pass(benchmarks), reverse=True)
    # Per-benchmark best (for the ★ winner marker).
    best_per_bench = {
        b: max((r.by_benchmark[b].pass_at_1 for r in rows if b in r.by_benchmark), default=0.0)
        for b in benchmarks
    }

    head = f"{'recipe':<16}{'kind':>8}"
    for b in benchmarks:
        head += f"{b[:10]:>12}"
    head += f"{'avg':>9}{'$/task':>10}"
    lines = [
        f"available models: {', '.join(available) or '(none)'}",
        f"tasks: {', '.join(benchmarks)}",
        "",
        head,
        "-" * len(head),
    ]
    for r in ranked:
        line = f"{r.recipe:<16}{'model' if r.is_model else 'RECIPE':>8}"
        for b in benchmarks:
            if b in r.by_benchmark:
                p = r.by_benchmark[b].pass_at_1
                star = "*" if abs(p - best_per_bench[b]) < 1e-9 else " "
                line += f"{p:>10.0%}{star:>2}"
            else:
                line += f"{'—':>12}"
        line += f"{r.avg_pass(benchmarks):>8.0%}{r.avg_cost(benchmarks):>10.5f}"
        lines.append(line)

    # Headline: best autoFusion recipe vs best single model.
    recipes_only = [r for r in rows if not r.is_model]
    models_only = [r for r in rows if r.is_model]
    best_recipe = max(recipes_only, key=lambda r: r.avg_pass(benchmarks), default=None)
    best_model = max(models_only, key=lambda r: r.avg_pass(benchmarks), default=None)
    lines += ["", "headline (avg across tasks):"]
    if best_model:
        lines.append(f"  best single model    : {best_model.recipe} ({best_model.avg_pass(benchmarks):.0%})")
    if best_recipe:
        lines.append(f"  best autoFusion recipe: {best_recipe.recipe} ({best_recipe.avg_pass(benchmarks):.0%})")
    if best_recipe and best_model:
        delta = (best_recipe.avg_pass(benchmarks) - best_model.avg_pass(benchmarks)) * 100
        verdict = "beats" if delta > 0 else ("ties" if abs(delta) < 1e-9 else "trails")
        lines.append(f"  -> our best recipe {verdict} the best single model by {delta:+.1f} pts")
    return "\n".join(lines)
