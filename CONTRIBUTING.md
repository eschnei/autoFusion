# Contributing to autoFusion

Thanks for helping build an honest, self-hostable fusion harness. The bar is simple: **changes must be measured, not asserted.**

## Dev setup

```bash
uv sync                                  # install (incl. dev deps)
ollama serve & ; ollama pull llama3.2     # local $0 models for live runs
uv run pytest                             # run the suite
```

No API keys are needed to develop — everything runs against local Ollama, and the test suite is model-free (it monkeypatches the provider) except the benchmark calibration tests, which only need the dataset.

## The golden rule: never weaken the eval

The eval harness is the project's thermometer. A green checkmark that isn't backed by a *correct* measurement is worse than a failure.

- `tests/test_scoring.py` (HumanEval) and `tests/test_livecodebench.py` calibrate the scorers against **known-correct solutions**. These **must** stay green. If your change makes them fail, the scorer is wrong — fix the scorer, do not relax the test.
- When you add a benchmark, you **must** add a calibration test that runs a known-correct solution through your scorer and asserts PASS.
- Never silently cap coverage (top-N, sampling, skipped cases). `log()`/`print()` what was dropped — silent truncation reads as "covered everything" when it didn't.

## How to extend

- **Add a model** — a `[[models]]` entry in your config (litellm id, optional `api_base`, per-token cost; `0` = free/local). No code.
- **Add a benchmark** — implement the `Benchmark` interface in `src/autofusion/eval/benchmarks.py` (`load()` → `Task`s, `score()` → `ScoreResult`), register it in `BENCHMARKS`, and add a **calibration test**.
- **Add a strategy** — implement `.run(messages, budget=None, **kw) -> CompletionResult` (see `SingleModel`/`Fusion`/`Router` in `strategies.py`) and wire it into `resolve_strategy`. It's then scored by the same eval as everything else.

## Style

- Python 3.11+, stdlib `argparse` CLI, dataclasses, async via litellm `acomplete`.
- Pin new dependencies. Keep provider normalization in LiteLLM — don't hand-write adapters.
- **Never hardcode or commit API keys.** Read them from env / `.env` only.

## Tests + PRs

Run `uv run pytest` (all green) before opening a PR. Describe what you measured, not just what you changed.
