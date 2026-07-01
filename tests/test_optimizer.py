"""Recipe optimizer (MAR-27), model-free."""

from autofusion.config import (
    BudgetConfig, CascadeConfig, Config, FusionConfig, ModelSpec, RouterConfig,
)
from autofusion.optimizer import (
    RecipeOutcome, available_model_names, candidate_recipes, mark_pareto, model_available, recommend,
)


def _local(name):
    return ModelSpec(name=name, model=f"ollama/{name}", input_cost_per_token=0.0,
                     output_cost_per_token=0.0)


def _hosted(name, model):
    return ModelSpec(name=name, model=model)


def test_local_always_available_hosted_needs_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert model_available(_local("llama3.2")) is True
    assert model_available(_hosted("gpt", "gpt-4o")) is False
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert model_available(_hosted("gpt", "gpt-4o")) is True


def _config():
    return Config(
        models={"local": _local("local"), "gpt": _hosted("gpt", "gpt-4o")},
        fusion=FusionConfig(proposers=["local", "gpt"], aggregator="gpt", layers=1),
        budget=BudgetConfig(), router=RouterConfig(), cascade=CascadeConfig(),
    )


def test_candidate_recipes_only_use_available(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = _config()
    available = available_model_names(cfg)
    assert available == ["local"]                       # gpt has no key
    recipes = candidate_recipes(cfg, available)
    assert recipes == ["local"]                         # fusion needs gpt -> excluded


def test_fusion_recipe_appears_when_all_models_available(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = _config()
    available = available_model_names(cfg)
    assert set(available) == {"local", "gpt"}
    assert "fusionMarj" in candidate_recipes(cfg, available)


def _outcome(name, q, cost):
    return RecipeOutcome(recipe=name, pass_at_1=q, avg_cost_usd=cost, avg_calls=1, avg_latency_s=1)


def test_pareto_drops_dominated_recipes():
    a = _outcome("A", 0.90, 0.10)   # best quality
    b = _outcome("B", 0.80, 0.01)   # cheapest
    c = _outcome("C", 0.85, 0.10)   # dominated by A (worse quality, same cost)
    mark_pareto([a, b, c])
    assert a.on_frontier and b.on_frontier
    assert not c.on_frontier


def test_recommend_picks_best_quality_and_cheapest_frontier():
    a = _outcome("A", 0.90, 0.10)
    b = _outcome("B", 0.80, 0.01)
    outs = mark_pareto([a, b])
    rec = recommend(outs)
    assert rec["best_quality"].recipe == "A"
    assert rec["cheapest_on_frontier"].recipe == "B"
