# Phase 3 — Calibration findings

Reproduce with: `uv run python scripts/phase3_calibrate.py -n 40`

*HumanEval, first 40 tasks · local Ollama ($0) · 2026-06-29*

| strategy | pass@1 | calls/q | avg lat(s) | Δ vs best single |
|---|---|---|---|---|
| self-moa | **95.0%** (38/40) | 4.0 | 38.9 | +2.5 pts |
| qwen2.5-coder *(aggregator-alone)* | 92.5% (37/40) | 1.0 | 9.1 | — (best single) |
| fusion-mixed | 90.0% (36/40) | 4.0 | 36.0 | −2.5 pts |
| llama3.2 | 75.0% (30/40) | 1.0 | 4.8 | −17.5 pts |
| qwen2.5 | 75.0% (30/40) | 1.0 | 4.5 | −17.5 pts |

- **self-moa** = 3 temperature-0.7 samples of qwen2.5-coder, aggregated by qwen2.5-coder.
- **fusion-mixed** = llama3.2 + qwen2.5 + qwen2.5-coder proposers, aggregated by qwen2.5-coder.

## What this reproduces

Both published findings showed up directionally on our own instrument:

1. **Aggregator paradox** — `fusion-mixed` (90.0%) scored **below** the aggregator
   running alone (92.5%). Mixing in two weaker 75% proposers *dragged the strong
   aggregator down*, at 4× the calls. Paying more for less.
2. **Self-MoA > mixed-MoA** ([arXiv 2502.00674](https://arxiv.org/abs/2502.00674)) —
   ensembling diverse samples of the *single best* model (95.0%) beat mixing
   different models (90.0%).

## The honest caveat (do not over-claim)

self-moa's "win" over aggregator-alone is **+2.5 points = exactly one extra
problem** (38 vs 37 of 40). At n=40 that is **within noise**, and self-moa is
non-deterministic (temp 0.7), so a rerun could erase it. Meanwhile it cost
**4.3× the latency** (38.9s vs 9.1s per query). So the defensible reading is:

> At this scale, self-MoA ≈ the best single model, and mixed-MoA is worse than
> the best single model. No config delivered a gain large enough to clearly
> justify its cost.

To firm this up: scale to the full 164 tasks with multiple self-moa seeds.

## Why this is the product, not a disappointment

Fusion was a free win in one config and a net loss in another — and the **only
way to know which** was to measure. That is exactly autoFusion's thesis: *don't
assert that fusion is better; prove it per task and per config, with the cost
attached.* The eval harness earned its place as Phase 1.
