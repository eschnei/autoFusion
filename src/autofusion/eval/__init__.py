"""Phase 1 — the eval harness (the thermometer).

A benchmark is: load tasks -> build a prompt -> run a model -> score
deterministically -> aggregate. The scorer interface is pluggable so new
verifiable benchmarks (GSM8K, ...) drop in without touching the runner.
"""
