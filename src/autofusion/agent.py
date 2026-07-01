"""Agentic coding — the single-agent loop (v2 Phase A).

A provider-agnostic ReAct loop over LiteLLM tool-calling: the model reads/edits
files and runs commands in a workspace until it calls `finish` or hits the step/
budget cap. This is the primitive best-of-N trajectories (Phase B) will run N of.

SECURITY: the `run` tool executes model-generated shell commands in the
workspace, and file edits are confined to it (path traversal rejected) but bash
is not. Point `--repo` at a repo you'd let a coding agent operate in — ideally a
disposable checkout or a container. A hardened sandbox is a later phase.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import litellm

from .budget import BudgetTracker
from .config import ModelSpec
from .providers import _litellm_kwargs

litellm.drop_params = True

AGENT_SYSTEM = (
    "You are a precise coding agent working inside a repository. Use the tools to "
    "read files, make focused edits, and run commands/tests until the task is done "
    "and any tests pass. Prefer `edit_file` for small changes. Run the tests to "
    "verify your work. Call `finish` only when the task is complete. Be efficient — "
    "don't read files you don't need."
)

TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a file in the repo.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Create or overwrite a file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "edit_file",
        "description": "Replace an exact substring (must occur exactly once) in a file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old": {"type": "string"},
                       "new": {"type": "string"}}, "required": ["path", "old", "new"]}}},
    {"type": "function", "function": {"name": "run", "description": "Run a shell command in the repo (tests, build, grep).",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "finish", "description": "Signal the task is complete.",
        "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": []}}},
]


class Workspace:
    """File + shell tools confined to a root directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def _resolve(self, path: str) -> Path:
        p = (self.root / path).resolve()
        if self.root != p and self.root not in p.parents:
            raise ValueError(f"path escapes workspace: {path}")
        return p

    def read_file(self, path: str) -> str:
        return self._resolve(path).read_text()[:8000]

    def write_file(self, path: str, content: str) -> str:
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"wrote {path} ({len(content)} bytes)"

    def edit_file(self, path: str, old: str, new: str) -> str:
        p = self._resolve(path)
        text = p.read_text()
        n = text.count(old)
        if n != 1:
            return f"error: `old` must match exactly once (found {n} occurrences)"
        p.write_text(text.replace(old, new))
        return f"edited {path}"

    def run(self, command: str, timeout: float = 60.0) -> str:
        try:
            proc = subprocess.run(command, shell=True, cwd=self.root,
                                  capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return f"timeout after {timeout}s"
        out = (proc.stdout + proc.stderr).strip()
        return f"exit {proc.returncode}\n{out[-3500:]}"

    def tree(self, max_files: int = 60) -> str:
        files = [str(p.relative_to(self.root)) for p in sorted(self.root.rglob("*"))
                 if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts]
        return "\n".join(files[:max_files])


def execute_tool(ws: Workspace, name: str, args: dict) -> str:
    try:
        if name == "read_file":
            return ws.read_file(args["path"])
        if name == "write_file":
            return ws.write_file(args["path"], args["content"])
        if name == "edit_file":
            return ws.edit_file(args["path"], args["old"], args["new"])
        if name == "run":
            return ws.run(args["command"])
        return f"error: unknown tool {name}"
    except Exception as exc:  # noqa: BLE001 — tool errors go back to the model, not up
        return f"error: {exc}"


@dataclass
class AgentResult:
    summary: str
    cost_usd: float
    steps: int
    n_calls: int
    finished: bool
    error: str | None = None


def _assistant_dict(msg) -> dict:
    d = {"role": "assistant", "content": msg.content or ""}
    if getattr(msg, "tool_calls", None):
        d["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ]
    return d


def _parse_args(raw: str) -> dict:
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}


async def run_agent(
    spec: ModelSpec, task: str, workspace: Workspace, *,
    budget: BudgetTracker | None = None, max_steps: int = 20,
) -> AgentResult:
    prompt = f"{task}\n\nRepository files:\n{workspace.tree()}"
    messages = [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": prompt}]
    total_cost, n_calls, finished, summary = 0.0, 0, False, ""

    for step in range(1, max_steps + 1):
        if budget and budget.total_usd is not None and budget.spent_usd >= budget.total_usd:
            return AgentResult("budget cap reached", total_cost, step - 1, n_calls, finished,
                               error="budget cap reached")
        kwargs = _litellm_kwargs(spec, messages, tools=TOOL_SCHEMAS, tool_choice="auto")
        try:
            resp = await litellm.acompletion(**kwargs)
        except Exception as exc:  # noqa: BLE001
            return AgentResult(summary, total_cost, step, n_calls, finished, error=str(exc))
        n_calls += 1
        cost = (resp._hidden_params or {}).get("response_cost") or 0.0
        total_cost += cost
        if budget:
            budget.record(cost)

        msg = resp.choices[0].message
        messages.append(_assistant_dict(msg))
        if not getattr(msg, "tool_calls", None):
            return AgentResult(msg.content or "", total_cost, step, n_calls, True)  # answered, no tools

        for tc in msg.tool_calls:
            if tc.function.name == "finish":
                summary = _parse_args(tc.function.arguments).get("summary", "done")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": "finished"})
                finished = True
                break
            result = execute_tool(workspace, tc.function.name, _parse_args(tc.function.arguments))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)[:3500]})
        if finished:
            return AgentResult(summary, total_cost, step, n_calls, True)

    return AgentResult(summary or "max steps reached", total_cost, max_steps, n_calls, finished)
