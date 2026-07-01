# autoFusion — Agentic Coding (v2) Spec

*Status: design. Owner: Eric. Supersedes nothing; extends the single-shot v1 recipes to multi-turn, tool-using agents.*

---

## 1. Thesis

v1 proved that on **verifiable, single-shot** code, a cheap basket + **verify-and-select** (`bestofMarj`) reaches near-frontier quality at a fraction of the cost — because the tests pick the winner, not a model's judgment.

Real coding is **agentic**: read a repo, edit files, run commands/tests, iterate until it works. v2 asks: **do autoFusion's ideas (verify-and-select, ensembling, capability routing) lift from a single answer to a whole agent trajectory?**

The bet: **yes, and the lever is the same** — in an agent loop the verifier is *already there* (run the tests after the edits). So we can run several agent attempts and keep the one whose tests pass. The recipe moves from "best of N answers" to "best of N **trajectories**."

---

## 2. The gap

Everything in v1 is `run(messages) -> one answer`. An agent is a **loop**:

```
while not done and within budget:
    model decides an action (read / edit / run / search / finish)  ← tool call
    harness executes it, returns the result
    model sees the result, decides the next action
```

We don't have: (a) a tool-using loop, (b) tools (read/edit/bash/grep), (c) a repo-aware task surface, (d) an agentic benchmark to measure recipes on. That's the v2 build.

---

## 3. Architecture

### 3.1 The agent loop (the missing primitive)
A minimal, provider-agnostic ReAct loop over LiteLLM tool-calling:
- **Tools:** `read_file`, `write_file`, `edit` (string-replace), `run` (bash: tests, build, grep), `finish`. Reuse the sandbox (`eval/sandbox.py`) hardened further.
- **Loop:** model emits tool calls → harness executes → results appended → repeat until `finish` or step/token/$ budget hit. Budget caps (MAR-17) enforced per call.
- **Context:** the task, the repo tree, and the running transcript; compact when long.

### 3.2 Recipes as *agent* strategies (the autoFusion angle)
Same idea as v1, lifted to trajectories. Each is a `Strategy`-like unit scored by the same eval:

| Recipe | Agentic form | Cost | Notes |
|---|---|---|---|
| **single** | one model runs the loop | 1 trajectory | baseline |
| **bestofMarj-agent** ★ | run **N independent trajectories** (diverse models/temps), run the tests on each, keep the first that passes | N× loops | the natural fit — tests are the verifier, exactly our proven win |
| **cascade-agent** | cheap agent first; if its tests fail, escalate to a stronger agent | ~1–2× | the cost saver |
| **delegate (Devin-Fusion)** | one strong "lead" plans + reviews, delegates mechanical edits to a cheap "sidekick" within one trajectory | ~1× | the frontier-quality-at-lower-cost pattern; most complex |

`bestofMarj-agent` is the headline — it's `bestofMarj` with a whole coding session as the "candidate" and the repo's tests as the verifier.

### 3.3 The verifier
Per trajectory: run the repo's test command; **pass = tests green**. Same held-out discipline as MAR-42 — if the benchmark ships hidden tests, grade on those; the agent only sees the visible ones. This keeps `bestofMarj-agent` honest.

### 3.4 The agentic eval
Measure which agent recipe wins, per cost — "measure, don't assert," at the trajectory level.
- **v2a (local, cheap):** a small **local bug-fix suite** — each task = a self-contained repo + a failing test + the fix. No Docker; runs in the sandbox. Enough to compare recipes.
- **v2b (real, heavy):** **SWE-bench Verified** slice — real repos, Docker per instance, `FAIL_TO_PASS`/`PASS_TO_PASS` grading. This is the credible number but the big infra lift (Docker, ~GBs of images).

### 3.5 Productization
`autofusion agent "<task>" --repo <path> --tests "<cmd>" [--recipe bestofMarj-agent] [--n N]` — an agent that works in your repo, iterates against your tests, and (for best-of-N) returns the trajectory that passes, as a diff you review.

---

## 4. Phased plan

| Phase | Deliverable | Gate |
|---|---|---|
| **A — single agent loop** | tool-using loop + tools + `agent` command; works in a repo against a test command | fixes a real bug in a sample repo, tests green, budget-capped |
| **B — best-of-N agents** ★ | run N trajectories, verify by tests, keep the winner; the recipe where our thesis is tested agentically | on the local bug-fix suite, best-of-N beats single-agent at measured cost |
| **C — local agentic eval** | the local bug-fix suite + scorer; recipes scored by the same harness | leaderboard: single vs best-of-N vs cascade-agent, quality × cost |
| **D — delegation + SWE-bench** | lead+sidecar delegation; SWE-bench Verified slice (Docker) | a real, defensible SWE-bench number for our recipes vs a frontier agent |

Build A→B→C first (all local, no Docker). D is a separate, heavier effort.

---

## 5. Honest hard parts / risks

- **Cheap models are worse *agents* than they are *generators*.** The agentic capability gap between cheap-open and frontier is **wider** than the single-shot gap — tool-use discipline, long-horizon coherence, and not-getting-lost are where frontier models pull ahead most. So `bestofMarj-agent` over cheap models may **not** beat one frontier agent the way `bestofMarj` beat single-shot. Verify-and-select still helps, but base agentic competence dominates. **This must be measured, not assumed** — it may reshape the recipe toward "cheap sidekick + frontier lead" (delegation) rather than "all-cheap best-of-N."
- **Tool execution = real sandboxing.** The model runs bash/edits in a repo. Needs isolation (containers, no network, resource limits) beyond our current subprocess guard — especially for untrusted/parallel trajectories.
- **Provider tool-calling varies.** Anthropic/OpenAI have solid native tool-use; open models via OpenRouter are inconsistent at it. LiteLLM normalizes the shape but not the reliability. Expect to filter the basket to models that can actually drive a loop.
- **Cost/latency multiply.** An agent is many calls; best-of-N trajectories multiply again. Budget caps and small N are essential; the local eval keeps iteration cheap.
- **Scope.** Full agentic coding is Cursor/Devin-scale. v2 targets **bounded, test-guarded tasks in one repo** — not open-ended autonomy.
- **Build vs. buy.** A minimal LiteLLM-based loop keeps us provider-agnostic and lets us inject recipe logic; wrapping an existing agent framework is faster but couples us to it. Recommend: build the minimal loop (it's the point).

---

## 6. Recommendation

Scope v2 to **Phases A→B→C** (all local, no Docker):
1. **A** gives a genuinely useful tool immediately (an agent that fixes bounded bugs against your tests).
2. **B** is where autoFusion's thesis gets its agentic test — best-of-N trajectories, tests as verifier.
3. **C** measures it honestly on a local suite before spending on Docker/SWE-bench.

Defer **D** (delegation + SWE-bench Verified) to a second effort once A–C show whether cheap-basket agents are competitive — because the risk in §5 (cheap models weaker as agents) is real, and the local eval will tell us early and cheaply which recipe to invest in.

---

## 7. Open questions

- **A. Which basket can actually drive a loop?** Measure tool-use reliability per model before committing baskets (some cheap-open models can't).
- **B. Best-of-N vs delegation** — is the win "N cheap agents + verify" or "1 frontier lead + cheap sidekick"? §5 predicts delegation may win; C decides it with data.
- **C. Sandbox depth** — subprocess guard vs. container per trajectory. Parallel best-of-N likely forces containers.
- **D. Local suite design** — how many tasks, what difficulty, held-out tests, to be a trustworthy thermometer without Docker.
