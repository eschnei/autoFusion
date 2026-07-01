"""Phase C — local agentic eval + suite validity. Model-free."""

import asyncio
import os
import subprocess
import sys
from types import SimpleNamespace

import autofusion.agent as agentmod
from autofusion.config import ModelSpec
from autofusion.eval import agenteval
from autofusion.eval.agentbench import LOCAL_SUITE, get_suite


# --------------------------------------------------------------------------- #
# The suite must be a real thermometer: every task fails buggy, passes fixed.
# --------------------------------------------------------------------------- #

def _run(cmd, cwd):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}   # same guard the agent uses
    return subprocess.run(cmd, shell=True, cwd=cwd, env=env,
                          capture_output=True, text=True).returncode


def test_every_task_fails_buggy_and_passes_fixed(tmp_path):
    for task in LOCAL_SUITE:
        d = tmp_path / task.id
        d.mkdir()
        task.materialize(d)
        cmd = task.test_cmd.replace("python", sys.executable, 1)
        assert _run(cmd, d) != 0, f"{task.id}: buggy repo should FAIL its test"
        for rel, content in task.fix.items():          # apply the reference fix
            (d / rel).write_text(content)
        assert _run(cmd, d) == 0, f"{task.id}: fixed repo should PASS its test"


# --------------------------------------------------------------------------- #
# The runner + recipes (single / bestof / cascade), with a mocked agent.
# --------------------------------------------------------------------------- #

class _Cfg:
    """Minimal config stub for parse_recipe."""
    code = SimpleNamespace(models=["bad", "good"])
    bestofn = SimpleNamespace(models=[])
    cascade = SimpleNamespace(tiers=["cheap", "strong"])
    delegate = SimpleNamespace(lead="lead", sidekick="hands", max_rounds=3)

    def model(self, name):
        return ModelSpec(name=name, model=f"ollama/{name}",
                         input_cost_per_token=0.0, output_cost_per_token=0.0)


def _fake_agent(fixers, cost=0.01):
    """fixers: model name -> {path: content} it writes. Missing name => no-op (fails)."""
    async def fake(spec, task, ws, **kw):
        for rel, content in fixers.get(spec.name, {}).items():
            ws.write_file(rel, content)
        return agentmod.AgentResult("done", cost, 1, 1, True)
    return fake


def _eval(cfg, recipes, monkeypatch, fixers):
    fake = _fake_agent(fixers)
    monkeypatch.setattr(agentmod, "run_agent", fake)       # bestof/cascade call it inside agent.py
    monkeypatch.setattr(agenteval, "run_agent", fake)      # single calls agenteval's by-name import
    return asyncio.run(agenteval.run_agent_eval(cfg, recipes, limit=1, max_steps=3, concurrency=2))


def test_single_recipe_scores_pass_and_fail(monkeypatch):
    fix = get_suite()[0].fix                                   # arithmetic fix -> mathx.py
    res = _eval(_Cfg(), ["single:good", "single:bad"], monkeypatch, {"good": fix})
    by = {r.model: r for r in res}
    assert by["single:good"].n_passed == 1                    # good writes the fix
    assert by["single:bad"].n_passed == 0                     # bad no-ops -> test fails
    assert by["single:good"].outcomes[0].n_calls == 1


def test_bestof_recipe_picks_winner_in_basket(monkeypatch):
    fix = get_suite()[0].fix
    res = _eval(_Cfg(), ["bestof:2"], monkeypatch, {"good": fix})  # basket = [bad, good]
    assert res[0].n_passed == 1
    assert res[0].outcomes[0].n_calls == 2                    # both trajectories ran


def test_cascade_escalates_and_sums_cost(monkeypatch):
    fix = get_suite()[0].fix
    # cheap no-ops (fails) -> escalate to strong (fixes). Both tiers attempted.
    res = _eval(_Cfg(), ["cascade"], monkeypatch, {"strong": fix})
    o = res[0].outcomes[0]
    assert res[0].n_passed == 1
    assert o.n_calls == 2                                     # cheap + strong both tried
    assert abs(o.cost_usd - 0.02) < 1e-9


def test_cascade_stops_at_first_passing_tier(monkeypatch):
    fix = get_suite()[0].fix
    # cheap already fixes it -> strong must never run (cost of one tier only).
    res = _eval(_Cfg(), ["cascade"], monkeypatch, {"cheap": fix, "strong": fix})
    o = res[0].outcomes[0]
    assert res[0].n_passed == 1
    assert o.n_calls == 1 and abs(o.cost_usd - 0.01) < 1e-9   # short-circuited


def test_parse_explicit_baskets_and_tiers():
    cfg = _Cfg()
    bo = agenteval.parse_recipe(cfg, "bestof:2+deepseek+qwen+llama")   # inline basket overrides config
    assert bo.kind == "bestof" and bo.n == 2
    assert [s.name for s in bo.specs] == ["deepseek", "qwen", "llama"]
    ca = agenteval.parse_recipe(cfg, "cascade+haiku+opus")
    assert ca.kind == "cascade" and [s.name for s in ca.specs] == ["haiku", "opus"]
    dg = agenteval.parse_recipe(cfg, "delegate:opus+deepseek")
    assert dg.kind == "delegate" and [s.name for s in dg.specs] == ["opus", "deepseek"]
    dg2 = agenteval.parse_recipe(cfg, "delegate")                       # falls back to [delegate] config
    assert [s.name for s in dg2.specs] == ["lead", "hands"]


def _fake_lead(plan="Edit mathx.py to fix the operators.", cost=0.005):
    async def fake(**kwargs):
        msg = SimpleNamespace(content=plan, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                               _hidden_params={"response_cost": cost})
    return fake


def test_delegate_recipe_sums_lead_and_sidekick(monkeypatch):
    fix = get_suite()[0].fix
    # sidekick "hands" executes the fix; lead is a mocked planning call.
    monkeypatch.setattr(agentmod, "run_agent", _fake_agent({"hands": fix}, cost=0.01))
    monkeypatch.setattr(agentmod.litellm, "acompletion", _fake_lead(cost=0.005))
    res = asyncio.run(agenteval.run_agent_eval(_Cfg(), ["delegate"], limit=1, max_steps=3))
    o = res[0].outcomes[0]
    assert res[0].n_passed == 1
    assert o.n_calls == 2                                    # 1 lead plan + 1 sidekick loop
    assert abs(o.cost_usd - 0.015) < 1e-9                    # lead cost + sidekick cost


def test_unknown_recipe_is_a_clean_error(monkeypatch):
    monkeypatch.setattr(agentmod, "run_agent", _fake_agent({}))
    try:
        asyncio.run(agenteval.run_agent_eval(_Cfg(), ["frobnicate"], limit=1))
        assert False, "should have raised"
    except ValueError as exc:
        assert "unknown recipe" in str(exc)
