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
import time
from dataclasses import dataclass

from .config import Config, ModelSpec
from .providers import CompletionResult, Message, acomplete

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

    async def run(self, messages: list[Message], **kw) -> CompletionResult:
        return await acomplete(self.spec, messages, **kw)


@dataclass
class Fusion:
    """Mixture-of-Agents: `layers` of parallel proposers, then one aggregator."""

    proposers: list[ModelSpec]
    aggregator: ModelSpec
    layers: int = 1
    name: str = "fusion"

    async def run(self, messages: list[Message], **kw) -> CompletionResult:
        start = time.perf_counter()
        total_cost = 0.0
        n_calls = 0
        candidates: list[str] = []

        for layer in range(self.layers):
            # Layer 0 proposers see the raw task; later layers see prior drafts.
            layer_messages = (
                messages if layer == 0 else build_aggregate_messages(messages, candidates)
            )
            results = await asyncio.gather(
                *(acomplete(p, layer_messages, **kw) for p in self.proposers)
            )
            total_cost += sum(r.cost_usd for r in results)
            n_calls += len(results)
            candidates = [r.text for r in results if r.ok]
            if not candidates:
                return CompletionResult(
                    self.name, "", total_cost, time.perf_counter() - start, 0, 0,
                    error="all proposers failed", n_calls=n_calls,
                )

        agg = await acomplete(self.aggregator, build_aggregate_messages(messages, candidates), **kw)
        total_cost += agg.cost_usd
        n_calls += 1
        return CompletionResult(
            model=self.name, text=agg.text, cost_usd=total_cost,
            latency_s=time.perf_counter() - start,
            prompt_tokens=agg.prompt_tokens, completion_tokens=agg.completion_tokens,
            error=agg.error, n_calls=n_calls,
        )


def resolve_strategy(config: Config, name: str):
    """Map a CLI name to a Strategy. A configured model -> SingleModel;
    the literal "fusion" -> Fusion built from the [fusion] config block."""
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
    known = ", ".join(sorted(config.models)) + ", fusion"
    raise KeyError(f"unknown strategy '{name}'. Available: {known}")
