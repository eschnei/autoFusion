"""Local agentic eval (Phase C) — 'measure, don't assert' at the trajectory level.

Runs agent recipes (single / best-of-N / cascade) over the local bug-fix suite
and scores them with the SAME instrument as the model benchmarks: pass@1 (the
fraction of bugs fixed) annotated with cost and calls-per-task. This is the
honest thermometer for §5 of the spec — do cheap-basket agent recipes actually
compete with a single frontier agent, and at what cost?

Recipe tokens:
  single:<model>   one agent trajectory with that model
  bestof:<N>       N trajectories over the coding basket; tests pick the winner
  cascade          escalating cascade over [cascade].tiers (cheap first)
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from ..agent import Workspace, best_of_n_agents, cascade_agents, run_agent
from ..budget import BudgetTracker
from ..config import ModelSpec
from .agentbench import AgentTask, get_suite
from .results import ModelRunResult, TaskOutcome


@dataclass
class Recipe:
    name: str          # the token, used as the leaderboard label
    kind: str          # single | bestof | cascade
    specs: list[ModelSpec]
    n: int = 1


def parse_recipe(cfg, token: str) -> Recipe:
    """Turn a recipe token into a Recipe, drawing baskets/tiers from the config."""
    if token.startswith("single:"):
        return Recipe(token, "single", [cfg.model(token.split(":", 1)[1])])
    if token.startswith("bestof"):
        n = int(token.replace("bestof", "").lstrip(":") or "3")
        names = cfg.code.models or cfg.bestofn.models
        if not names:
            raise ValueError("bestof recipe needs [code].models or [bestofn].models")
        return Recipe(token, "bestof", [cfg.model(m) for m in names], n=n)
    if token == "cascade":
        if not cfg.cascade.tiers:
            raise ValueError("cascade recipe needs [cascade].tiers")
        return Recipe(token, "cascade", [cfg.model(m) for m in cfg.cascade.tiers])
    raise ValueError(f"unknown recipe '{token}' (use single:<model>, bestof:<N>, or cascade)")


async def _run_one(recipe: Recipe, task: AgentTask, budget, max_steps: int) -> TaskOutcome:
    tmp = Path(tempfile.mkdtemp(prefix="af-ceval-"))
    task.materialize(tmp)
    t0 = time.perf_counter()
    try:
        if recipe.kind == "single":
            ws = Workspace(tmp)
            res = await run_agent(recipe.specs[0], task.prompt, ws, budget=budget, max_steps=max_steps)
            passed = (await asyncio.to_thread(ws.run, task.test_cmd)).startswith("exit 0")
            cost, calls, err = res.cost_usd, res.n_calls, res.error
        else:  # bestof / cascade both return a BestOfResult over `tmp`
            fn = best_of_n_agents if recipe.kind == "bestof" else cascade_agents
            kw = {"budget": budget, "max_steps": max_steps}
            bo = (await fn(recipe.specs, task.prompt, tmp, task.test_cmd, recipe.n, **kw)
                  if recipe.kind == "bestof"
                  else await fn(recipe.specs, task.prompt, tmp, task.test_cmd, **kw))
            passed = bo.winner is not None
            cost = bo.total_cost
            calls = sum(t.result.n_calls for t in bo.trajectories)
            err = None
        return TaskOutcome(
            task_id=task.id, model=recipe.name, passed=passed,
            detail=(recipe.specs[0].name if recipe.kind == "single" else recipe.kind),
            cost_usd=cost, latency_s=time.perf_counter() - t0, n_calls=calls, error=err,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _aggregate(name: str, outcomes: list[TaskOutcome]) -> ModelRunResult:
    return ModelRunResult(
        model=name, benchmark="local-bugfix", n_tasks=len(outcomes),
        n_passed=sum(o.passed for o in outcomes),
        n_errors=sum(o.error is not None for o in outcomes),
        total_cost_usd=sum(o.cost_usd for o in outcomes),
        avg_latency_s=(sum(o.latency_s for o in outcomes) / len(outcomes)) if outcomes else 0.0,
        total_calls=sum(o.n_calls for o in outcomes),
        outcomes=outcomes,
    )


async def run_agent_eval(
    cfg, recipe_tokens: list[str], *, suite: str = "local", limit: int | None = None,
    budget: BudgetTracker | None = None, max_steps: int = 15, concurrency: int = 4,
) -> list[ModelRunResult]:
    tasks = get_suite(suite)
    if limit:
        tasks = tasks[:limit]
    recipes = [parse_recipe(cfg, t) for t in recipe_tokens]  # parse up front so a typo fails fast
    results: list[ModelRunResult] = []
    for recipe in recipes:  # recipes sequential (shared budget); tasks concurrent within a recipe
        sem = asyncio.Semaphore(concurrency)

        async def guarded(task: AgentTask, r: Recipe = recipe) -> TaskOutcome:
            async with sem:
                return await _run_one(r, task, budget, max_steps)

        outcomes = await asyncio.gather(*[guarded(t) for t in tasks])
        results.append(_aggregate(recipe.name, outcomes))
    return results
