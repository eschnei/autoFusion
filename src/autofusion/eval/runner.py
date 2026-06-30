"""Baseline runner (Phase 1) — now strategy-aware (Phase 2).

Runs each strategy (single model OR fusion) across a benchmark and records
score + cost + calls + latency. Fusion is measured by the exact same
instrument as any single-model baseline — that's the whole point.
"""

from __future__ import annotations

import asyncio

from ..budget import BudgetTracker
from ..config import Config
from ..strategies import resolve_strategy
from .benchmarks import Task, get_benchmark
from .results import ModelRunResult, TaskOutcome


async def _run_one(strategy, task: Task, benchmark, sem: asyncio.Semaphore, budget) -> TaskOutcome:
    # Strategies that select among candidates (best-of-N) get the real verifier:
    # the benchmark's own scorer for this task. Others never see it.
    extra = {}
    if getattr(strategy, "needs_verifier", False):
        extra["verify"] = lambda text, _t=task: benchmark.score(_t, text).passed
    async with sem:
        completion = await strategy.run(task.messages, budget=budget, temperature=0.0, **extra)
    if not completion.ok:
        return TaskOutcome(
            task_id=task.task_id, model=strategy.name, passed=False,
            detail=f"completion error: {completion.error}", cost_usd=completion.cost_usd,
            latency_s=completion.latency_s, n_calls=completion.n_calls, error=completion.error,
        )
    # Scoring (subprocess execution) is sync/CPU-bound — offload off the loop.
    score = await asyncio.to_thread(benchmark.score, task, completion.text)
    return TaskOutcome(
        task_id=task.task_id, model=strategy.name, passed=score.passed, detail=score.detail,
        cost_usd=completion.cost_usd, latency_s=completion.latency_s, n_calls=completion.n_calls,
    )


async def run_strategy(
    strategy, tasks: list[Task], benchmark, concurrency: int = 4, budget=None
) -> ModelRunResult:
    sem = asyncio.Semaphore(concurrency)
    outcomes = await asyncio.gather(*(_run_one(strategy, t, benchmark, sem, budget) for t in tasks))
    n = len(outcomes)
    return ModelRunResult(
        model=strategy.name, benchmark=benchmark.name, n_tasks=n,
        n_passed=sum(o.passed for o in outcomes),
        n_errors=sum(o.error is not None for o in outcomes),
        total_cost_usd=sum(o.cost_usd for o in outcomes),
        total_calls=sum(o.n_calls for o in outcomes),
        avg_latency_s=sum(o.latency_s for o in outcomes) / n if n else 0.0,
        outcomes=list(outcomes),
    )


async def run_baseline(
    config: Config, names: list[str], benchmark_name: str,
    limit: int | None = None, concurrency: int = 4, budget: BudgetTracker | None = None,
) -> list[ModelRunResult]:
    benchmark = get_benchmark(benchmark_name)
    tasks = benchmark.load(limit=limit)
    # Reuse a caller-supplied tracker so a multi-benchmark report shares ONE cap;
    # otherwise the cap would reset per benchmark.
    if budget is None:
        budget = BudgetTracker.from_config(config.budget)
    results = []
    for name in names:  # one strategy at a time keeps a local Ollama from thrashing
        strategy = resolve_strategy(config, name)
        results.append(
            await run_strategy(strategy, tasks, benchmark, concurrency=concurrency, budget=budget)
        )
    return results
