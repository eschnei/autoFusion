"""Baseline runner (Phase 1).

Runs each single model in the pool across a benchmark and records score + cost
+ latency. This is the instrument every later strategy (fusion, router) is
measured against — they plug in here as just another "model".
"""

from __future__ import annotations

import asyncio

from ..config import Config
from ..providers import acomplete
from .benchmarks import Task, get_benchmark
from .results import ModelRunResult, TaskOutcome


async def _run_one(spec, task: Task, benchmark, sem: asyncio.Semaphore) -> TaskOutcome:
    async with sem:
        completion = await acomplete(spec, task.messages, temperature=0.0)
    if not completion.ok:
        return TaskOutcome(
            task_id=task.task_id, model=spec.name, passed=False,
            detail=f"completion error: {completion.error}", cost_usd=completion.cost_usd,
            latency_s=completion.latency_s, error=completion.error,
        )
    # Scoring (subprocess execution) is sync/CPU-bound — offload off the loop.
    score = await asyncio.to_thread(benchmark.score, task, completion.text)
    return TaskOutcome(
        task_id=task.task_id, model=spec.name, passed=score.passed, detail=score.detail,
        cost_usd=completion.cost_usd, latency_s=completion.latency_s,
    )


async def run_model(spec, tasks: list[Task], benchmark, concurrency: int = 4) -> ModelRunResult:
    sem = asyncio.Semaphore(concurrency)
    outcomes = await asyncio.gather(*(_run_one(spec, t, benchmark, sem) for t in tasks))
    n = len(outcomes)
    n_passed = sum(o.passed for o in outcomes)
    n_errors = sum(o.error is not None for o in outcomes)
    total_cost = sum(o.cost_usd for o in outcomes)
    avg_lat = sum(o.latency_s for o in outcomes) / n if n else 0.0
    return ModelRunResult(
        model=spec.name, benchmark=benchmark.name, n_tasks=n, n_passed=n_passed,
        n_errors=n_errors, total_cost_usd=total_cost, avg_latency_s=avg_lat,
        outcomes=list(outcomes),
    )


async def run_baseline(
    config: Config, model_names: list[str], benchmark_name: str,
    limit: int | None = None, concurrency: int = 4,
) -> list[ModelRunResult]:
    benchmark = get_benchmark(benchmark_name)
    tasks = benchmark.load(limit=limit)
    results = []
    for name in model_names:  # one model at a time keeps a local Ollama from thrashing
        spec = config.model(name)
        results.append(await run_model(spec, tasks, benchmark, concurrency=concurrency))
    return results
