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
    name: str = "fusionMarj"
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
    name: str = "routeMarj"

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


_CRITIC_SYSTEM = (
    "You are a strict grader. Given a TASK and a CANDIDATE ANSWER, estimate the "
    "probability the answer is correct and complete. Respond with ONLY a single "
    "number between 0 and 1 (e.g. 0.85) — no words."
)


async def _critic_score(
    critic: ModelSpec, messages: list[Message], answer: str, budget: BudgetTracker | None, **kw
) -> tuple[float | None, CompletionResult]:
    """Cheap critic rates a candidate 0..1. Returns (score|None, its result).
    None = unparseable/failed → caller should fail-safe to escalation."""
    task_text = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
    critic_messages = [
        {"role": "system", "content": _CRITIC_SYSTEM},
        {"role": "user", "content": f"TASK:\n{task_text}\n\nCANDIDATE ANSWER:\n{answer}\n\nProbability correct (0-1):"},
    ]
    result = await _guarded_acomplete(critic, critic_messages, budget, temperature=0.0)
    if not result.ok:
        return None, result
    match = re.search(r"\d*\.?\d+", result.text)
    if not match:
        return None, result
    return max(0.0, min(1.0, float(match.group()))), result


@dataclass
class Cascade:
    """Cost cascade: run `tiers` cheapest-first; after each non-final tier a
    `critic` scores the answer and we stop early if confident, else escalate.
    Most requests resolve at the cheap tier, so the pricey tiers (and fusion)
    are paid for only on the hard tail."""

    tiers: list  # Strategy objects, cheapest -> most expensive
    critic: ModelSpec
    threshold: float = 0.7
    name: str = "cascadeMarj"

    async def run(
        self, messages: list[Message], budget: BudgetTracker | None = None, **kw
    ) -> CompletionResult:
        start = time.perf_counter()
        total_cost = 0.0
        n_calls = 0
        last: CompletionResult | None = None

        for i, tier in enumerate(self.tiers):
            result = await tier.run(messages, budget=budget, **kw)
            total_cost += result.cost_usd
            n_calls += result.n_calls
            last = result
            if i == len(self.tiers) - 1:
                break  # final tier — return whatever it gave
            if not result.ok:
                continue  # tier errored — escalate
            score, crit = await _critic_score(self.critic, messages, result.text, budget, **kw)
            total_cost += crit.cost_usd
            n_calls += crit.n_calls
            if score is not None and score >= self.threshold:
                break  # confident enough — stop here, don't pay for higher tiers
            # else: low confidence or unparseable critic -> escalate (fail-safe)

        return CompletionResult(
            model=self.name, text=last.text, cost_usd=total_cost,
            latency_s=time.perf_counter() - start,
            prompt_tokens=last.prompt_tokens, completion_tokens=last.completion_tokens,
            error=last.error, n_calls=n_calls,
        )


@dataclass
class VerifiedBestOfN:
    """Sample N candidates from a basket of models, then let a VERIFIER pick the
    winner — the cheap path to high quality on verifiable tasks (no frontier
    judge needed). The runner passes a real `verify(text)` (the benchmark scorer)
    because `needs_verifier` is set. With no verifier, a `critic` model picks;
    failing that, the first candidate."""

    models: list[ModelSpec]
    n: int = 4
    critic: ModelSpec | None = None
    temperature: float = 0.7
    name: str = "bestofMarj"
    needs_verifier: bool = True  # signals the runner to pass a verify() closure

    async def run(
        self, messages: list[Message], budget: BudgetTracker | None = None,
        verify=None, **kw,
    ) -> CompletionResult:
        start = time.perf_counter()
        kw.pop("temperature", None)  # we set sampling temperature ourselves
        # Generate N candidates, cycling the basket for cross-model diversity.
        gens = [
            _guarded_acomplete(self.models[i % len(self.models)], messages, budget,
                               temperature=self.temperature, **kw)
            for i in range(self.n)
        ]
        results = await asyncio.gather(*gens)
        total_cost = sum(r.cost_usd for r in results)
        n_calls = len(results)
        oks = [r for r in results if r.ok]

        def finish(chosen: CompletionResult | None, error: str | None = None) -> CompletionResult:
            return CompletionResult(
                model=self.name, text=chosen.text if chosen else "", cost_usd=total_cost,
                latency_s=time.perf_counter() - start,
                prompt_tokens=chosen.prompt_tokens if chosen else 0,
                completion_tokens=chosen.completion_tokens if chosen else 0,
                error=error, n_calls=n_calls,
            )

        if not oks:
            return finish(None, error="all candidates failed")

        if verify is not None:
            for r in oks:
                if await asyncio.to_thread(verify, r.text):
                    return finish(r)  # first verified-correct candidate wins
            # none verified -> fall through to critic / first (honest: likely wrong)

        if self.critic is not None:
            best, best_score = oks[0], -1.0
            for r in oks:
                score, crit = await _critic_score(self.critic, messages, r.text, budget)
                total_cost += crit.cost_usd
                n_calls += crit.n_calls
                if (score or 0.0) > best_score:
                    best, best_score = r, (score or 0.0)
            return finish(best)

        return finish(oks[0])


@dataclass
class CategoryRouter:
    """Classify a task by category (regex over the prompt) and dispatch to that
    category's sub-strategy — a model OR a recipe (bestofMarj for code, opus for
    reasoning, ...). Passes a held-out verifier through only to sub-strategies
    that select among candidates."""

    default: object  # a Strategy
    rules: list[tuple[re.Pattern, object]] = field(default_factory=list)  # (pattern, Strategy)
    name: str = "auto"
    needs_verifier: bool = True

    def select(self, messages: list[Message]):
        text = " ".join(m.get("content", "") for m in messages if m.get("role") == "user")
        for pattern, strat in self.rules:
            if pattern.search(text):
                return strat
        return self.default

    async def run(
        self, messages: list[Message], budget: BudgetTracker | None = None,
        verify=None, **kw,
    ) -> CompletionResult:
        sub = self.select(messages)
        if getattr(sub, "needs_verifier", False):
            return await sub.run(messages, budget=budget, verify=verify, **kw)
        return await sub.run(messages, budget=budget, **kw)


# Our recipes are branded "*Marj". Old bare names stay as aliases so existing
# configs/commands keep working.
_ALIASES = {"fusion": "fusionMarj", "route": "routeMarj",
            "cascade": "cascadeMarj", "bestofn": "bestofMarj"}


def resolve_strategy(config: Config, name: str):
    """Map a CLI name to a Strategy. A configured model -> SingleModel;
    "fusionMarj" -> Fusion from [fusion]; "routeMarj" -> Router from [router];
    "cascadeMarj" -> Cascade; "bestofMarj" -> VerifiedBestOfN. Bare legacy names
    (fusion/route/cascade/bestofn) resolve as aliases."""
    if name in config.models:
        return SingleModel(config.model(name))
    name = _ALIASES.get(name, name)
    if name == "fusionMarj":
        f = config.fusion
        if not f.proposers or not f.aggregator:
            raise ValueError("[fusion] config needs both proposers and an aggregator")
        return Fusion(
            proposers=[config.model(p) for p in f.proposers],
            aggregator=config.model(f.aggregator),
            layers=f.layers,
        )
    if name == "routeMarj":
        r = config.router
        if not r.default:
            raise ValueError("[router] config needs a default model")
        return Router(
            default=config.model(r.default),
            rules=[(re.compile(p, re.IGNORECASE), config.model(m)) for p, m in r.rules],
        )
    if name == "cascadeMarj":
        c = config.cascade
        if len(c.tiers) < 2 or not c.critic:
            raise ValueError("[cascade] config needs >=2 tiers and a critic model")
        if {"cascade", "cascadeMarj"} & set(c.tiers):
            raise ValueError("[cascade] tiers cannot include the cascade itself")
        return Cascade(
            tiers=[resolve_strategy(config, t) for t in c.tiers],
            critic=config.model(c.critic),
            threshold=c.threshold,
        )
    if name == "bestofMarj":
        b = config.bestofn
        if not b.models:
            raise ValueError("[bestofn] config needs at least one model")
        return VerifiedBestOfN(
            models=[config.model(m) for m in b.models],
            n=b.n,
            critic=config.model(b.critic) if b.critic else None,
            temperature=b.temperature,
        )
    if name == "auto":
        from .recipe_cache import load_recipes

        cats = config.categories
        if not cats.default:
            raise ValueError("[categories] config needs a default strategy")
        learned = load_recipes(config)  # {category: {"recipe": ...}}

        def pick(category: str, fallback: str) -> str:
            recipe = learned.get(category, {}).get("recipe", fallback)
            return fallback if recipe == "auto" else recipe  # never self-route

        return CategoryRouter(
            default=resolve_strategy(config, pick("default", cats.default)),
            rules=[(re.compile(pat, re.IGNORECASE), resolve_strategy(config, pick(cat, strat)))
                   for cat, pat, strat in cats.rules],
        )
    known = ", ".join(sorted(config.models)) + ", fusionMarj, routeMarj, cascadeMarj, bestofMarj, auto"
    raise KeyError(f"unknown strategy '{name}'. Available: {known}")
