# Fusion Harness — Build Brief

*Working title. Open-source, developer-first multi-model fusion harness.*
*Status: pre-build. Owner: Eric. Last updated: build kickoff.*

---

## 1. Thesis

A single frontier model commits to one draw from its distribution and inherits its own blind spots. A **fusion harness** sends a task to several models in parallel (proposers), then has an aggregator synthesize their outputs into a final answer — a mixture-of-agents (MoA) architecture. On benchmarks, this can exceed the best individual model by a meaningful margin, at the cost of N× the calls.

We are building the open-source, self-hostable version of that. The code is free; the user brings their own API keys (or runs local models for a true $0 path). We prove the quality claim with our own eval harness instead of asserting it.

**What this is not (v1):** a router. Routing picks one model per task — cheaper, but bounded by the best single model, so it can only *match* frontier cheaply, never beat it. Routing is deferred to Phase 6.

---

## 2. Locked decisions

These are settled. Don't relitigate them mid-build.

| Decision | Choice | Why |
|---|---|---|
| Primary architecture | **Fusion (MoA) first**, router later | Fusion is the thing that beats frontier; router is bounded by best single model |
| Audience | **Developer-first**, delivered as a **CLI** | That's where people code; CLI is the native surface |
| Auth | **API keys only**, bring-your-own | Subscription-CLI auth is a ToS/ban risk (token-extraction enforcement, Jan 2026) and no longer cost-advantaged (programmatic metering, Jun 2026). Hard no for a tool others run. |
| License | **MIT or Apache 2.0** | Permissive, zero marginal cost, contributor-friendly |
| Provider plumbing | **LiteLLM** | Normalizes 100+ providers to one OpenAI-compatible interface; don't hand-write adapters |
| Free path | **Ollama as a first-class provider** | Local open-weight proposers = real $0 tier and the differentiator vs. hosted Fusion offerings |
| Build order | **Eval harness before orchestrator** | The eval is the thermometer; without it "fusion is better" is unfalsifiable |

---

## 3. Architecture at a glance

```
[ Interface ]        CLI  +  OpenAI-compatible HTTP endpoint
[ Orchestrator ]     fusion strategy (proposers → aggregator)  [router later]
[ Provider layer ]   LiteLLM → OpenAI / Anthropic / Google / Ollama (local)
[ Registry+Config ]  per-model cost/latency/capability  +  BYO keys  +  budget caps
[ Eval + Logging ]   benchmark runner, scoring, baseline comparison, cost tracking
```

The eval/logging layer is not an afterthought — it's built **second**, right after minimal plumbing, and everything else is validated through it.

---

## 4. Build phases

Each phase has a concrete **done-when** gate. Don't start the next phase until the gate is met.

### Phase 0 — Scaffold & single-model plumbing
The minimum needed to call one model end-to-end.

- [ ] Create repo, pick license (MIT/Apache), add README stub stating the thesis
- [ ] Set up Python project (uv/poetry), config loading from a local file + env vars
- [ ] Integrate LiteLLM; implement a `complete(model, messages)` wrapper
- [ ] Wire BYO keys: read provider keys from config/env, never hardcode
- [ ] Smoke test: call one hosted model and one Ollama local model through the same interface

**Done when:** one function calls any configured model (hosted or local) and returns a response.

### Phase 1 — Eval harness (the thermometer)
Build the measurement instrument before the thing being measured.

- [ ] **Pick a verifiable starter benchmark** (see Open Question A) — recommend code (HumanEval / small SWE-bench slice) or math (GSM8K)
- [ ] Implement a benchmark loader (tasks + ground truth)
- [ ] Implement deterministic scoring for that benchmark (run tests / exact-match the number) — **no LLM judge yet**
- [ ] Implement a **baseline runner**: run *each single model in the pool* across the benchmark and record score + cost + latency
- [ ] Implement result logging: per-task outcome, per-run cost total, per-model breakdown

**Done when:** you can run every single model against the benchmark and get a scored, cost-annotated leaderboard.

### Phase 2 — Minimal fusion (MoA)
The simplest possible fusion, nothing fancy.

- [ ] Implement proposer step: send the prompt to K models in parallel (async, with timeouts + retries)
- [ ] Implement aggregator step: one model receives all proposer outputs and synthesizes a final answer
- [ ] Make proposers, aggregator, and K configurable
- [ ] Run fusion through the **Phase 1 eval harness** unchanged

**Done when:** fusion runs as one more "model" the eval harness can score, same as any baseline.

### Phase 3 — Calibrate (reproduce a known win)
Prove the thermometer reads true before trusting any new number.

- [ ] Reproduce a **published** fusion/MoA result on a public benchmark within a reasonable margin
- [ ] Report the **paired metric**: quality delta **and** cost multiple together (e.g. "+4% vs. best single model, at 5× calls")
- [ ] **Include aggregator-alone as a baseline.** If fusion only ties the aggregator running solo, you're paying N× to use one model well — that's the aggregator paradox, and this is the test for it.

**Done when:** the harness reproduces a known fusion win, and the report shows delta + cost multiple + the aggregator-alone comparison. If it can't reproduce a known result, the code is wrong — fix before proceeding.

### Phase 4 — Make it usable
Turn the proof-of-concept into something a developer can actually run.

- [ ] **Cost guardrails:** per-request cost ceiling + global budget cap, enforced before calls fire (fusion can silently 5× a bill — this is a feature, not polish)
- [ ] **Local $0 path:** validate small/local proposers + one frontier aggregator as a documented config
- [ ] **CLI:** clean command surface (`fuse "prompt"`, config init, eval run, budget status)
- [ ] **OpenAI-compatible HTTP endpoint:** so existing tools/IDEs can point at the harness with one base-URL change
- [ ] Config UX: a `init` command that scaffolds the config and checks which keys are present

**Done when:** a developer can install, add their keys, and run fusion from the CLI or via the endpoint, with a budget cap respected.

### Phase 5 — Open-source readiness
Make it safe and inviting for strangers to run and contribute to.

- [ ] README: thesis, quickstart, the honest cost story (fusion = N× calls), the $0 local path
- [ ] Document the auth stance explicitly (API keys only; why no subscription auth) so nobody re-adds the ban risk
- [ ] Model **registry**: ship a starter set of model metadata (cost/latency/capability) and decide the maintenance model (see Open Question C)
- [ ] CONTRIBUTING + example configs + a reproducible eval others can run
- [ ] Tag v1.0

**Done when:** a new user can go from clone → first fusion call → first eval run using only the docs.

### Phase 6 — Router (deferred)
Only after fusion is proven and shipped.

- [ ] Add a routing strategy alongside fusion (swappable, same interface)
- [ ] Decide decision mechanism (see Open Question B): heuristics → trained classifier (RouteLLM-style) → LLM-judge
- [ ] Eval routing on the *same* harness: did it pick the model that would have scored highest?

**Done when:** users can choose `--strategy route` vs. `--strategy fuse`, both evaluated by the same instrument.

---

## 5. Open questions to resolve

These are still open. The plan can start without them, but each has a "decide by" point.

**A. Which verifiable benchmark first?** *(Decide by start of Phase 1.)*
Recommendation: **code** (HumanEval or a small SWE-bench slice) — auto-gradable, and the public fusion results you'll reproduce in Phase 3 used coding/agentic benchmarks. Math (GSM8K) is the simpler fallback. Pick verifiable regardless of your eventual product domain, so you can build before solving the judging problem.

**B. Router decision mechanism.** *(Decide by Phase 6, not before.)*
Heuristics (free, transparent, brittle) vs. trained classifier (needs data) vs. LLM-judge per call (flexible, adds cost/latency to every request). No free lunch — defer.

**C. Who keeps the registry current?** *(Decide by Phase 5.)*
Models and prices shift weekly. Options: hand-maintained (rots), community-contributed, or auto-pulled from provider APIs. Lean toward auto-pull + community override.

**D. The one marketing wedge.** *(Decide by Phase 5 README.)*
You can *support* cheaper, higher-ceiling, and private/local simultaneously. You can only *lead* with one. Given the Ollama path, "self-hosted, no markup, real $0 tier" is the strongest differentiator vs. hosted Fusion offerings — but it's your call.

**E. Open-ended scoring (later).** When developers throw non-verifiable work at the harness, you'll need an LLM-judge. Mitigations to design then: randomize answer order (position bias), use a *different* model family as judge than the aggregator (self-preference/circularity), watch for length bias.

---

## 6. Risks & how the plan handles them

| Risk | Mitigation in this plan |
|---|---|
| "Fusion isn't actually better, just more expensive" | Phase 3 forces aggregator-alone baseline + paired cost metric; the eval can falsify the whole premise early |
| Silent cost blowups erode trust | Phase 4 cost guardrails enforced pre-call |
| Auth approach gets someone banned | Locked to API keys; documented explicitly in Phase 5 |
| Benchmarks saturate / get gamed | Public benchmark only for calibration (Phase 3); durable value is a private domain eval set built later |
| Building the fun part first | Eval harness is Phase 1; orchestrator can't be trusted without it |

---

## 7. Definition of done (v1.0)

- A developer can install the CLI, add API keys (or run fully local via Ollama), and execute fusion.
- The harness reproduces a known public fusion benchmark result.
- Every eval reports quality delta **and** cost multiple, with aggregator-alone as a baseline.
- Budget caps are enforced.
- Docs let a stranger go clone → fusion call → eval run unaided.
- License is permissive; auth is API-keys-only and documented.
- Router is explicitly scoped as Phase 6, not shipped in v1.
