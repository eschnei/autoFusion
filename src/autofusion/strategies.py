"""Strategies (Phase 2).

A Strategy turns a prompt into one answer. The point of this abstraction is that
fusion is "just another model" to the eval harness:

  - SingleModel — one model, one call (the Phase 1 baseline).
  - Fusion (MoA) — K proposers draft in parallel, an aggregator synthesizes.

Both return a CompletionResult, so the runner and leaderboard treat them
identically. Cost and call-count accumulate across the whole fusion graph, which
is exactly the "quality delta AND cost multiple" the brief demands.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field

from .budget import BudgetTracker, estimate_request_cost
from .config import Config, ModelSpec
from .providers import CompletionResult, Message, acomplete


async def _guarded_acomplete(
    spec: ModelSpec, messages: list[Message], budget: BudgetTracker | None, **kw
) -> CompletionResult:
    """acomplete, but a budget cap is checked BEFORE the call fires and the
    actual cost is recorded after. Over-cap raises BudgetExceeded (no call)."""
    if budget is not None:
        budget.check(estimate_request_cost(spec, messages))
    result = await acomplete(spec, messages, **kw)
    if budget is not None and result.ok:
        budget.record(result.cost_usd)
    return result

# Synthesis prompt appended as a final user turn. We keep the ORIGINAL messages
# (including any output-format system instruction, e.g. "return only a code
# block") in place so the aggregator's answer stays in the task's required
# format — then add the candidates and ask for the single best answer.
_SYNTH = (
    "Several AI models independently produced candidate answers to the task above:\n\n"
    "{candidates}\n\n"
    "Critically review these candidates — they may contain mistakes or disagree. "
    "Then produce the single best final answer to the ORIGINAL task, in exactly the "
    "format the task requires. Do not mention the candidates or that you compared them."
)


def _format_candidates(candidates: list[str]) -> str:
    return "\n\n".join(f"[Candidate {i + 1}]\n{c}" for i, c in enumerate(candidates))


def build_aggregate_messages(messages: list[Message], candidates: list[str]) -> list[Message]:
    """Original prompt + a synthesis turn carrying the candidate answers."""
    synth = _SYNTH.format(candidates=_format_candidates(candidates))
    return list(messages) + [{"role": "user", "content": synth}]


@dataclass
class SingleModel:
    spec: ModelSpec

    @property
    def name(self) -> str:
        return self.spec.name

    async def run(
        self, messages: list[Message], budget: BudgetTracker | None = None, **kw
    ) -> CompletionResult:
        return await _guarded_acomplete(self.spec, messages, budget, **kw)


@dataclass
class Fusion:
    """Mixture-of-Agents: `layers` of parallel proposers, then one aggregator."""

    proposers: list[ModelSpec]
    aggregator: ModelSpec
    layers: int = 1
    name: str = "fusion"
    # Sampling temperature for proposers. None inherits the call's temperature.
    # Set > 0 to get diverse drafts — required for Self-MoA (same model repeated).
    proposer_temperature: float | None = None

    async def run(
        self, messages: list[Message], budget: BudgetTracker | None = None, **kw
    ) -> CompletionResult:
        start = time.perf_counter()
        total_cost = 0.0
        n_calls = 0
        candidates: list[str] = []

        prop_kw = dict(kw)
        if self.proposer_temperature is not None:
            prop_kw["temperature"] = self.proposer_temperature

        for layer in range(self.layers):
            # Layer 0 proposers see the raw task; later layers see prior drafts.
            layer_messages = (
                messages if layer == 0 else build_aggregate_messages(messages, candidates)
            )
            results = await asyncio.gather(
                *(_guarded_acomplete(p, layer_messages, budget, **prop_kw) for p in self.proposers)
            )
            total_cost += sum(r.cost_usd for r in results)
            n_calls += len(results)
            candidates = [r.text for r in results if r.ok]
            if not candidates:
                return CompletionResult(
                    self.name, "", total_cost, time.perf_counter() - start, 0, 0,
                    error="all proposers failed", n_calls=n_calls,
                )

        agg = await _guarded_acomplete(
            self.aggregator, build_aggregate_messages(messages, candidates), budget, **kw
        )
        total_cost += agg.cost_usd
        n_calls += 1
        return CompletionResult(
            model=self.name, text=agg.text, cost_usd=total_cost,
            latency_s=time.perf_counter() - start,
            prompt_tokens=agg.prompt_tokens, completion_tokens=agg.completion_tokens,
            error=agg.error, n_calls=n_calls,
        )


@dataclass
class Router:
    """Picks ONE model per request via ordered regex rules over the prompt, else a
    default. Heuristic (free, transparent); classifier/LLM-judge routing deferred.
    Bounded by the best single model — the cheap complement to fusion."""

    default: ModelSpec
    rules: list[tuple[re.Pattern, ModelSpec]] = field(default_factory=list)
    name: str = "route"

    def select(self, messages: list[Message]) -> ModelSpec:
        text = " ".join(m.get("content", "") for m in messages if m.get("role") == "user")
        for pattern, spec in self.rules:
            if pattern.search(text):
                return spec
        return self.default

    async def run(
        self, messages: list[Message], budget: BudgetTracker | None = None, **kw
    ) -> CompletionResult:
        # The result's `model` is the chosen model (handy for the `route` CLI);
        # the eval harness labels the run by this strategy's name instead.
        return await _guarded_acomplete(self.select(messages), messages, budget, **kw)


def resolve_strategy(config: Config, name: str):
    """Map a CLI name to a Strategy. A configured model -> SingleModel;
    "fusion" -> Fusion from [fusion]; "route" -> Router from [router]."""
    if name in config.models:
        return SingleModel(config.model(name))
    if name == "fusion":
        f = config.fusion
        if not f.proposers or not f.aggregator:
            raise ValueError("[fusion] config needs both proposers and an aggregator")
        return Fusion(
            proposers=[config.model(p) for p in f.proposers],
            aggregator=config.model(f.aggregator),
            layers=f.layers,
        )
    if name == "route":
        r = config.router
        if not r.default:
            raise ValueError("[router] config needs a default model")
        return Router(
            default=config.model(r.default),
            rules=[(re.compile(p, re.IGNORECASE), config.model(m)) for p, m in r.rules],
        )
    known = ", ".join(sorted(config.models)) + ", fusion, route"
    raise KeyError(f"unknown strategy '{name}'. Available: {known}")
