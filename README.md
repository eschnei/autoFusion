# autoFusion

> Open-source, developer-first **multi-model fusion harness** with a built-in eval harness that *proves* when fusion beats a single model — instead of asserting it.

## Thesis

A single frontier model commits to one draw from its distribution and inherits its own blind spots. A **fusion harness** sends a task to several models in parallel (**proposers**), then has an **aggregator** synthesize their outputs into a final answer — a Mixture-of-Agents (MoA) architecture. On benchmarks this can exceed the best individual model, at the cost of N× the calls.

autoFusion is the self-hostable version of that. **The code is free; you bring your own API keys** — or run fully local via [Ollama](https://ollama.com) for a true **$0 path**. We don't claim fusion is better; the built-in eval harness lets you measure it on your own tasks, including against the **aggregator running alone** (the honest baseline).

**Not a router.** A router picks *one* model per task — cheaper, but bounded by the best single model. Fusion runs several and synthesizes. Routing is deferred to a later phase.

## The honest cost story

Fusion is **N+1 calls per request** (N proposers + 1 aggregation). It can silently multiply a bill, so budget caps are first-class. Recent research ([Self-MoA](https://arxiv.org/abs/2502.00674)) also found mixing models sometimes *loses* to sampling one good model, and that much of MoA's gain comes from the aggregator. autoFusion's stance: **don't trust the claim — run the eval.** That's why the eval harness is built before the fusion orchestrator.

## Status

Early build. Implemented:

- **Phase 0** — config + provider plumbing via [LiteLLM](https://github.com/BerriAI/litellm) (100+ providers + local Ollama through one interface).
- **Phase 1** — eval harness ("the thermometer"): HumanEval loader, deterministic sandboxed pass@1 scoring, per-model baseline runner with cost + latency, leaderboard.

Next: Phase 2 minimal fusion (MoA), Phase 3 calibration, Phase 4 CLI/budget/endpoint. See [`fusion-harness-build-brief.md`](fusion-harness-build-brief.md).

## Quickstart

```bash
uv sync                                     # install
ollama serve & ; ollama pull llama3.2       # local $0 path — no key needed
ollama pull qwen2.5:3b                       # a second proposer (for fusion)

uv run autofusion init                       # scaffold autofusion.toml + show key/endpoint status
uv run autofusion config-check               # which models/keys are usable
uv run autofusion smoke -m llama3.2          # call one model end-to-end
uv run autofusion fuse "your prompt"         # run fusion (MoA) on one prompt
uv run autofusion eval -m llama3.2,fusion -n 5   # score baselines + fusion on HumanEval
uv run autofusion budget status              # show the configured cost caps
uv run autofusion serve                      # OpenAI-compatible endpoint on :8000
```

Point any OpenAI client at `http://localhost:8000/v1` and use `model: "fusion"` (or a configured model name). All commands accept `-c/--config <path>`; for hosted models, `cp .env.example .env` and add a key. Budget caps in `[budget]` are enforced **before** any call fires.

Edit [`autofusion.toml`](autofusion.toml) to register models (litellm id, optional `api_base`, per-token cost; `0` = free/local).

### Local proposers + frontier aggregator (the cost sweet spot)

Phase 3 found fusion's gains hinge on a **strong aggregator**, not on the proposers. So the highest-value config drafts with cheap **local** Ollama models and synthesizes with **one hosted frontier model** — you pay for exactly **one** strong call per request while the N proposer drafts cost nothing:

```bash
# needs only OPENAI_API_KEY in .env — proposers are local
autofusion -c configs/local-plus-frontier.toml fuse "your prompt"
```

See [`configs/local-plus-frontier.toml`](configs/local-plus-frontier.toml). If the aggregator's key is missing, `config-check` flags it (`MISSING OPENAI_API_KEY`) rather than crashing, and the tight `[budget]` cap is your safety net since only the aggregator spends.

## Security note

HumanEval grading **executes model-generated code**. autoFusion runs each program in an isolated subprocess with a hard timeout, CPU/memory/file-size limits, and a reliability guard that neuters destructive syscalls (`src/autofusion/eval/sandbox.py`). This is adequate for benchmark models you control — **not** a boundary for adversarial code. For untrusted-at-scale use, run inside a locked-down container (no network, gVisor/seccomp).

## License

Apache 2.0. **Auth is API-keys-only by design** — subscription/CLI token auth is a ToS/ban risk for a tool others run, and is intentionally not supported.
