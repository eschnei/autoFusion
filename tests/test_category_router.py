"""CategoryRouter — route tasks to per-category recipes (MAR-44), model-free."""

import asyncio
import re

from autofusion.config import (
    BestOfNConfig, BudgetConfig, CategoriesConfig, Config, FusionConfig, ModelSpec, RouterConfig,
)
from autofusion.providers import CompletionResult
from autofusion.strategies import CategoryRouter, resolve_strategy


class _Stub:
    def __init__(self, name, needs_verifier=False):
        self.name = name
        self.needs_verifier = needs_verifier
        self.got_verify = "unset"

    async def run(self, messages, budget=None, verify="MISSING", **kw):
        self.got_verify = verify
        return CompletionResult(self.name, f"from {self.name}", 0.0, 0.0, 1, 1)


def _router():
    code, math, default = _Stub("bestofMarj", needs_verifier=True), _Stub("math"), _Stub("opus")
    return CategoryRouter(
        default=default,
        rules=[(re.compile("code|def ", re.I), code), (re.compile("math|equation", re.I), math)],
    )


def test_classifies_to_category():
    r = _router()
    assert r.select([{"role": "user", "content": "write code to sort"}]).name == "bestofMarj"
    assert r.select([{"role": "user", "content": "solve this equation"}]).name == "math"
    assert r.select([{"role": "user", "content": "write me a poem"}]).name == "opus"


def test_verify_passed_only_to_needs_verifier_sub():
    r = _router()
    code_sub = r.rules[0][1]
    math_sub = r.rules[1][1]
    asyncio.run(r.run([{"role": "user", "content": "write code"}], verify="VERIFIER"))
    assert code_sub.got_verify == "VERIFIER"          # needs_verifier -> receives it
    asyncio.run(r.run([{"role": "user", "content": "an equation"}], verify="VERIFIER"))
    assert math_sub.got_verify == "MISSING"           # not needs_verifier -> default (unset)


def test_resolve_strategy_builds_category_router():
    cfg = Config(
        models={n: ModelSpec(name=n, model=f"ollama/{n}", input_cost_per_token=0.0,
                             output_cost_per_token=0.0) for n in ("opus", "a", "b", "critic")},
        fusion=FusionConfig(), budget=BudgetConfig(), router=RouterConfig(),
        bestofn=BestOfNConfig(models=["a", "b"], critic="critic"),
        categories=CategoriesConfig(default="opus", rules=[("code|def ", "bestofMarj")]),
    )
    r = resolve_strategy(cfg, "auto")
    assert isinstance(r, CategoryRouter)
    assert r.default.name == "opus"
    assert r.rules[0][1].name == "bestofMarj"          # rule resolved to the recipe
