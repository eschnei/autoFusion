"""Learned recipe cache (MAR-45), model-free."""

from autofusion.config import (
    BestOfNConfig, BudgetConfig, CategoriesConfig, Config, FusionConfig, ModelSpec, RouterConfig,
)
from autofusion.recipe_cache import cache_path, load_recipes, save_recipes
from autofusion.strategies import resolve_strategy


def _spec(n):
    return ModelSpec(name=n, model=f"ollama/{n}", input_cost_per_token=0.0, output_cost_per_token=0.0)


def _cfg(tmp_path, cat_rules):
    return Config(
        models={n: _spec(n) for n in ("opus", "a", "b", "critic")},
        fusion=FusionConfig(), budget=BudgetConfig(), router=RouterConfig(),
        bestofn=BestOfNConfig(models=["a", "b"], critic="critic"),
        categories=CategoriesConfig(default="opus", rules=cat_rules),
        path=tmp_path / "autofusion.toml",
    )


def test_cache_round_trip(tmp_path):
    cfg = _cfg(tmp_path, [])
    assert load_recipes(cfg) == {}                    # none yet
    data = {"code": {"recipe": "bestofMarj", "score": 0.9, "cost": 0.001, "benchmark": "lcb", "n": 10}}
    p = save_recipes(cfg, data)
    assert p == cache_path(cfg) and p.exists()
    assert load_recipes(cfg) == data


def test_auto_uses_cached_recipe_over_config(tmp_path):
    # config says code -> bestofMarj; the learned cache overrides it to opus.
    cfg = _cfg(tmp_path, [("code", "code|def ", "bestofMarj")])
    assert resolve_strategy(cfg, "auto").rules[0][1].name == "bestofMarj"   # no cache -> config
    save_recipes(cfg, {"code": {"recipe": "opus"}})
    assert resolve_strategy(cfg, "auto").rules[0][1].name == "opus"          # cache wins


def test_cache_ignores_self_route(tmp_path):
    cfg = _cfg(tmp_path, [("code", "code", "bestofMarj")])
    save_recipes(cfg, {"code": {"recipe": "auto"}})    # would self-route
    assert resolve_strategy(cfg, "auto").rules[0][1].name == "bestofMarj"    # falls back to config
