"""Result records + logging + leaderboard rendering (Phase 1)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class TaskOutcome:
    task_id: str
    model: str
    passed: bool
    detail: str
    cost_usd: float
    latency_s: float
    n_calls: int = 1
    error: str | None = None


@dataclass
class ModelRunResult:
    """One model's score across a benchmark — the unit the leaderboard ranks.

    `model` is treated as just a name, so a fusion strategy logs here too,
    scored by the same instrument as any single-model baseline.
    """

    model: str
    benchmark: str
    n_tasks: int
    n_passed: int
    n_errors: int
    total_cost_usd: float
    avg_latency_s: float
    total_calls: int = 0
    outcomes: list[TaskOutcome] = field(default_factory=list)

    @property
    def pass_at_1(self) -> float:
        return self.n_passed / self.n_tasks if self.n_tasks else 0.0

    @property
    def avg_calls(self) -> float:
        return self.total_calls / self.n_tasks if self.n_tasks else 0.0


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021). For n=1 this is just c."""
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod


def save_run(results: list[ModelRunResult], out_dir: str | Path = "results") -> Path:
    """Write per-task JSONL + a summary JSON for a benchmark run."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bench = results[0].benchmark if results else "run"
    base = out / f"{stamp}_{bench}"

    with (base.with_suffix(".jsonl")).open("w") as fh:
        for r in results:
            for o in r.outcomes:
                fh.write(json.dumps(asdict(o)) + "\n")

    summary = [
        {
            "model": r.model,
            "benchmark": r.benchmark,
            "pass_at_1": round(r.pass_at_1, 4),
            "n_passed": r.n_passed,
            "n_tasks": r.n_tasks,
            "n_errors": r.n_errors,
            "total_cost_usd": round(r.total_cost_usd, 6),
            "avg_calls": round(r.avg_calls, 2),
            "avg_latency_s": round(r.avg_latency_s, 3),
        }
        for r in results
    ]
    summary_path = base.with_name(base.name + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary_path


def render_leaderboard(results: list[ModelRunResult]) -> str:
    """ASCII leaderboard, ranked by pass@1, annotated with cost + latency."""
    ranked = sorted(results, key=lambda r: r.pass_at_1, reverse=True)
    header = (
        f"{'model':<22}{'pass@1':>9}{'passed':>9}{'errors':>8}"
        f"{'cost($)':>11}{'calls/q':>9}{'avg lat(s)':>12}"
    )
    lines = [header, "-" * len(header)]
    for r in ranked:
        lines.append(
            f"{r.model:<22}{r.pass_at_1:>8.1%}{r.n_passed:>4}/{r.n_tasks:<4}"
            f"{r.n_errors:>8}{r.total_cost_usd:>11.4f}{r.avg_calls:>9.1f}{r.avg_latency_s:>12.2f}"
        )
    return "\n".join(lines)
