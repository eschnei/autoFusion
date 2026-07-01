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
