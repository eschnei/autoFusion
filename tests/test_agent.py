"""Agent loop + tools (MAR-46 Phase A), model-free."""

import asyncio
import json
from types import SimpleNamespace

import autofusion.agent as agentmod
from autofusion.agent import Workspace, execute_tool, run_agent
from autofusion.config import ModelSpec


def _spec():
    return ModelSpec(name="m", model="ollama/m", input_cost_per_token=0.0, output_cost_per_token=0.0)


# ---- Workspace tools ----

def test_write_read_edit(tmp_path):
    ws = Workspace(tmp_path)
    assert "wrote" in ws.write_file("a.py", "x = 1\n")
    assert ws.read_file("a.py") == "x = 1\n"
    assert "edited" in ws.edit_file("a.py", "x = 1", "x = 2")
    assert ws.read_file("a.py") == "x = 2\n"


def test_edit_requires_unique_match(tmp_path):
    ws = Workspace(tmp_path)
    ws.write_file("a.py", "x\nx\n")
    assert "exactly once" in ws.edit_file("a.py", "x", "y")   # 2 matches -> refuse


def test_path_confinement_rejected(tmp_path):
    ws = Workspace(tmp_path)
    assert execute_tool(ws, "read_file", {"path": "../../etc/hosts"}).startswith("error")


def test_run_captures_exit_and_output(tmp_path):
    ws = Workspace(tmp_path)
    out = ws.run("echo hello")
    assert out.startswith("exit 0") and "hello" in out


# ---- The loop (mocked model) ----

def _tc(id, name, args):
    return SimpleNamespace(id=id, function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def _resp(content=None, tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)], _hidden_params={"response_cost": 0.002})


def _scripted(responses):
    it = iter(responses)

    async def fake(**kwargs):
        return next(it)

    return fake


def test_loop_executes_tools_then_finishes(tmp_path, monkeypatch):
    responses = [
        _resp(tool_calls=[_tc("1", "write_file", {"path": "sol.py", "content": "VALUE = 42\n"})]),
        _resp(tool_calls=[_tc("2", "finish", {"summary": "wrote sol.py"})]),
    ]
    monkeypatch.setattr(agentmod.litellm, "acompletion", _scripted(responses))
    result = asyncio.run(run_agent(_spec(), "create sol.py", Workspace(tmp_path), max_steps=5))
    assert result.finished and result.summary == "wrote sol.py"
    assert result.n_calls == 2 and abs(result.cost_usd - 0.004) < 1e-9
    assert (tmp_path / "sol.py").read_text() == "VALUE = 42\n"


def test_loop_stops_at_max_steps(tmp_path, monkeypatch):
    # model keeps calling a tool, never finishes -> capped
    looping = _resp(tool_calls=[_tc("x", "run", {"command": "true"})])
    monkeypatch.setattr(agentmod.litellm, "acompletion",
                        lambda **kw: _async_return(looping))
    result = asyncio.run(run_agent(_spec(), "loop forever", Workspace(tmp_path), max_steps=3))
    assert not result.finished and result.steps == 3 and result.n_calls == 3


async def _async_return(v):
    return v


# ---- Phase B: best-of-N trajectories ----

def _named(name):
    return ModelSpec(name=name, model=f"ollama/{name}", input_cost_per_token=0.0, output_cost_per_token=0.0)


def _repo_with_failing_test(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sol.py").write_text("def f():\n    return 0\n")
    (repo / "t.py").write_text("import sol\nassert sol.f() == 1\n")
    return repo


def test_best_of_applies_passing_winner(tmp_path, monkeypatch):
    repo = _repo_with_failing_test(tmp_path)

    async def fake_run_agent(spec, task, ws, **kw):  # 'good' writes the fix; others don't
        ws.write_file("sol.py", "def f():\n    return 1\n" if spec.name == "good"
                      else "def f():\n    return 9\n")
        return agentmod.AgentResult("done", 0.01, 1, 1, True)

    monkeypatch.setattr(agentmod, "run_agent", fake_run_agent)
    res = asyncio.run(agentmod.best_of_n_agents(
        [_named("bad"), _named("good")], "fix f", str(repo), "python t.py", 2, max_steps=5))
    assert res.winner is not None and res.winner.model == "good"
    assert "return 1" in (repo / "sol.py").read_text()      # winner applied to the real repo
    assert abs(res.total_cost - 0.02) < 1e-9                 # cost summed across trajectories


def test_best_of_none_pass_leaves_repo_unchanged(tmp_path, monkeypatch):
    repo = _repo_with_failing_test(tmp_path)

    async def fake_run_agent(spec, task, ws, **kw):
        ws.write_file("sol.py", "def f():\n    return 9\n")   # everyone fails
        return agentmod.AgentResult("done", 0.01, 1, 1, True)

    monkeypatch.setattr(agentmod, "run_agent", fake_run_agent)
    res = asyncio.run(agentmod.best_of_n_agents(
        [_named("a"), _named("b")], "fix f", str(repo), "python t.py", 2, max_steps=5))
    assert res.winner is None
    assert "return 0" in (repo / "sol.py").read_text()       # untouched
