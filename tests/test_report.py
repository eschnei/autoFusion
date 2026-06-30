"""Cross-task comparison report (MAR-30), model-free."""

from autofusion.optimizer import RecipeOutcome
from autofusion.report import ReportRow, render_report


def _row(name, is_model, scores):
    by = {b: RecipeOutcome(recipe=name, pass_at_1=p, avg_cost_usd=c, avg_calls=1, avg_latency_s=1)
          for b, (p, c) in scores.items()}
    return ReportRow(recipe=name, is_model=is_model, by_benchmark=by)


BENCH = ["humaneval", "gsm8k"]


def test_avg_pass_and_cost_across_tasks():
    r = _row("x", False, {"humaneval": (1.0, 0.0), "gsm8k": (0.6, 0.02)})
    assert r.avg_pass(BENCH) == 0.8
    assert abs(r.avg_cost(BENCH) - 0.01) < 1e-9


def test_render_picks_best_recipe_and_model():
    rows = [
        _row("gpt", True, {"humaneval": (0.9, 0.05), "gsm8k": (0.9, 0.05)}),     # best model 0.90
        _row("llama", True, {"humaneval": (0.7, 0.0), "gsm8k": (0.6, 0.0)}),
        _row("bestofn", False, {"humaneval": (1.0, 0.0), "gsm8k": (0.95, 0.0)}),  # best recipe 0.975
        _row("fusion", False, {"humaneval": (0.8, 0.0), "gsm8k": (0.8, 0.0)}),
    ]
    out = render_report(rows, BENCH, available=["llama", "gpt"])
    assert "best single model    : gpt (90%)" in out
    assert "best autoFusion recipe: bestofn (98%)" in out
    assert "beats the best single model by +7.5 pts" in out
    # the model/recipe split is shown
    assert "RECIPE" in out and "model" in out


def test_missing_benchmark_renders_dash():
    rows = [_row("x", False, {"humaneval": (1.0, 0.0)})]  # no gsm8k entry
    out = render_report(rows, BENCH, available=["x"])
    assert "—" in out
