"""Phase 3 — calibration.

Runs single-model baselines and several fusion configs against the SAME
HumanEval slice and reports the paired metric the brief demands: quality delta
AND call/cost multiple, with the aggregator-alone baseline made explicit.

This is the test that can falsify the whole premise: if no fusion config beats
the best single model (which IS the aggregator running alone), you're paying N×
to use one model well — the aggregator paradox.

Usage:
    uv run python scripts/phase3_calibrate.py [--limit N] [--concurrency K]
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path

from autofusion.config import load_config
from autofusion.eval.benchmarks import get_benchmark
from autofusion.eval.results import ModelRunResult, render_leaderboard
from autofusion.eval.runner import run_strategy
from autofusion.strategies import Fusion, SingleModel


def build_configs(cfg):
    """Strategies to compare. Aggregator for every fusion is qwen2.5-coder, so
    its single-model row below IS the aggregator-alone baseline."""
    m = cfg.model
    return [
        SingleModel(m("llama3.2")),
        SingleModel(m("qwen2.5")),
        SingleModel(m("qwen2.5-coder")),  # <- aggregator-alone baseline
        Fusion(
            name="fusion-mixed",
            proposers=[m("llama3.2"), m("qwen2.5"), m("qwen2.5-coder")],
            aggregator=m("qwen2.5-coder"),
            layers=1,
        ),
        Fusion(
            name="self-moa",  # 3 samples of the strong model, then it aggregates
            proposers=[m("qwen2.5-coder"), m("qwen2.5-coder"), m("qwen2.5-coder")],
            aggregator=m("qwen2.5-coder"),
            layers=1,
            proposer_temperature=0.7,
        ),
    ]


AGGREGATOR_ALONE = "qwen2.5-coder"


def write_report(results: list[ModelRunResult], limit: int, path: Path) -> None:
    ranked = sorted(results, key=lambda r: r.pass_at_1, reverse=True)
    singles = [r for r in results if r.total_calls == r.n_tasks]  # 1 call/task
    best_single = max(singles, key=lambda r: r.pass_at_1) if singles else None
    agg_alone = next((r for r in results if r.model == AGGREGATOR_ALONE), None)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# Phase 3 — Calibration report",
        "",
        f"*HumanEval, first {limit} tasks · local Ollama ($0) · {stamp}*",
        "",
        "Quality with the cost it actually took. `calls/q` is the fusion tax: "
        "fusion is only worth it if pass@1 rises **more** than calls/q does.",
        "",
        "| strategy | pass@1 | calls/q | avg lat(s) | Δ vs best single |",
        "|---|---|---|---|---|",
    ]
    for r in ranked:
        delta = ""
        if best_single is not None:
            d = (r.pass_at_1 - best_single.pass_at_1) * 100
            delta = "— (best single)" if r is best_single else f"{d:+.1f} pts"
        lines.append(
            f"| {r.model} | {r.pass_at_1:.1%} | {r.avg_calls:.1f} | "
            f"{r.avg_latency_s:.2f} | {delta} |"
        )

    lines += ["", "## Verdict", ""]
    if best_single and agg_alone:
        fusions = [r for r in results if r.total_calls > r.n_tasks]
        winners = [f for f in fusions if f.pass_at_1 > best_single.pass_at_1]
        if winners:
            w = max(winners, key=lambda r: r.pass_at_1)
            lines.append(
                f"- **Fusion won.** `{w.model}` at {w.pass_at_1:.1%} beat the best single "
                f"model `{best_single.model}` ({best_single.pass_at_1:.1%}) by "
                f"{(w.pass_at_1 - best_single.pass_at_1) * 100:+.1f} pts, at {w.avg_calls:.1f}× calls."
            )
        else:
            lines.append(
                f"- **Aggregator paradox confirmed (this config/scale).** No fusion config beat "
                f"the best single model `{best_single.model}` ({best_single.pass_at_1:.1%}). "
                f"Aggregator-alone (`{agg_alone.model}`, {agg_alone.pass_at_1:.1%}) matches or "
                f"beats fusion while costing 1 call instead of {max(f.avg_calls for f in fusions):.0f}."
            )
    lines += [
        "",
        f"*Caveats: n={limit}, small local models, greedy aggregation. A demonstration "
        "of the instrument, not a general verdict on fusion. Scale n and try stronger/more "
        "diverse proposers before drawing conclusions.*",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--limit", type=int, default=40)
    ap.add_argument("--concurrency", type=int, default=3)
    args = ap.parse_args()

    cfg = load_config()
    benchmark = get_benchmark("humaneval")
    tasks = benchmark.load(limit=args.limit)
    strategies = build_configs(cfg)

    results: list[ModelRunResult] = []
    for s in strategies:
        print(f"running: {s.name} ...", flush=True)
        r = asyncio.run(run_strategy(s, tasks, benchmark, concurrency=args.concurrency))
        print(f"  {s.name}: {r.pass_at_1:.1%} ({r.n_passed}/{r.n_tasks}), "
              f"{r.avg_calls:.1f} calls/q, {r.avg_latency_s:.1f}s avg", flush=True)
        results.append(r)

    print("\n" + render_leaderboard(results))
    out = Path("results")
    out.mkdir(exist_ok=True)
    report = out / "phase3_report.md"
    write_report(results, args.limit, report)
    print(f"\nreport: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
