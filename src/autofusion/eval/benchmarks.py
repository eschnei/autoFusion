"""Benchmark abstraction + HumanEval (Phase 1).

A Benchmark turns dataset rows into Tasks (a prompt to send a model) and scores
a model's raw text output deterministically. No LLM judge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .sandbox import run_python

Message = dict[str, str]


@dataclass
class Task:
    task_id: str
    messages: list[Message]  # what we send the model
    meta: dict = field(default_factory=dict)  # benchmark-specific grading data


@dataclass
class ScoreResult:
    passed: bool
    detail: str


def extract_code(text: str) -> str:
    """Pull a Python code block out of a chat model's reply.

    Prefers a fenced block containing a `def`; falls back to the longest fenced
    block, then to the raw text.
    """
    blocks = re.findall(r"```(?:python|py)?\s*\n?(.*?)```", text, re.DOTALL)
    if blocks:
        for b in blocks:
            if "def " in b:
                return b.strip()
        return max(blocks, key=len).strip()
    return text.strip()


class HumanEval:
    """OpenAI HumanEval — 164 problems, scored by executing unit tests (pass@1)."""

    name = "humaneval"

    _SYSTEM = (
        "You are an expert Python programmer. Implement the requested function. "
        "Return ONLY the complete function definition (including the signature) "
        "inside a single ```python code block. No explanations, no examples, no extra text."
    )

    def load(self, limit: int | None = None) -> list[Task]:
        from datasets import load_dataset

        rows = load_dataset("openai/openai_humaneval")["test"]
        if limit is not None:
            rows = rows.select(range(min(limit, len(rows))))
        tasks: list[Task] = []
        for row in rows:
            tasks.append(
                Task(
                    task_id=row["task_id"],
                    messages=[
                        {"role": "system", "content": self._SYSTEM},
                        {"role": "user", "content": row["prompt"]},
                    ],
                    meta={
                        "test": row["test"],
                        "entry_point": row["entry_point"],
                        "prompt": row["prompt"],
                    },
                )
            )
        return tasks

    def score(self, task: Task, output: str) -> ScoreResult:
        code = extract_code(output)
        entry = task.meta["entry_point"]
        if f"def {entry}" not in code:
            return ScoreResult(False, f"failed: no `def {entry}` in output")
        # HumanEval prompts declare imports (e.g. `from typing import List`) ABOVE
        # the signature. Chat models often omit them, so prepend the prompt's
        # preamble (everything before the function def) to the program. Otherwise
        # we'd penalize correct code for a missing import the task already supplied.
        prompt = task.meta.get("prompt", "")
        idx = prompt.find(f"def {entry}")
        preamble = prompt[:idx] if idx != -1 else ""
        program = "\n".join([preamble, code, "", task.meta["test"], "", f"check({entry})", ""])
        result = run_python(program, timeout=10.0)
        return ScoreResult(result.passed, result.detail)


BENCHMARKS = {HumanEval.name: HumanEval}


def get_benchmark(name: str):
    try:
        return BENCHMARKS[name]()
    except KeyError:
        known = ", ".join(sorted(BENCHMARKS))
        raise KeyError(f"unknown benchmark '{name}'. Available: {known}") from None
