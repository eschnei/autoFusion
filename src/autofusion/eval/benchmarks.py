"""Benchmark abstraction + HumanEval (Phase 1).

A Benchmark turns dataset rows into Tasks (a prompt to send a model) and scores
a model's raw text output deterministically. No LLM judge.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .sandbox import run_python, run_with_stdin

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


def _normalize_output(s: str) -> str:
    """Trailing-whitespace-insensitive stdout comparison."""
    return "\n".join(line.rstrip() for line in s.strip().splitlines())


class LiveCodeBench:
    """LiveCodeBench (code_generation_lite) — contamination-resistant competitive
    programming. Frontier models score ~50-70%, so unlike HumanEval there's real
    headroom to detect a fusion gain.

    Loaded directly from the repo's test*.jsonl (the HF loading script is
    incompatible with datasets 3.x). v1 grades stdin/stdout problems on their
    PUBLIC test cases; functional (LeetCode-style) problems are skipped and the
    count is logged — comparative validity holds since every strategy faces the
    identical tests.
    """

    name = "livecodebench"

    _SYSTEM = (
        "You are an expert competitive programmer. Write a COMPLETE Python 3 program "
        "that reads from standard input and writes the answer to standard output. "
        "Return ONLY the program in a single ```python code block — no explanation."
    )

    def load(self, limit: int | None = None, data_file: str = "test.jsonl") -> list[Task]:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download("livecodebench/code_generation_lite", data_file, repo_type="dataset")
        tasks: list[Task] = []
        skipped = 0
        with open(path) as fh:
            for line in fh:
                rec = json.loads(line)
                public = json.loads(rec["public_test_cases"])
                stdin_tests = [t for t in public if t.get("testtype") == "stdin"]
                if not stdin_tests:  # functional/LeetCode — out of scope for v1
                    skipped += 1
                    continue
                tasks.append(
                    Task(
                        task_id=rec["question_id"],
                        messages=[
                            {"role": "system", "content": self._SYSTEM},
                            {"role": "user", "content": rec["question_content"]},
                        ],
                        meta={"tests": stdin_tests, "difficulty": rec.get("difficulty")},
                    )
                )
                if limit is not None and len(tasks) >= limit:
                    break
        if skipped:
            print(f"[livecodebench] skipped {skipped} non-stdin (functional) problems")
        return tasks

    def score(self, task: Task, output: str) -> ScoreResult:
        code = extract_code(output)
        for i, test in enumerate(task.meta["tests"]):
            result = run_with_stdin(code, test["input"], timeout=10.0)
            if result.timed_out:
                return ScoreResult(False, f"timeout on test {i}")
            if _normalize_output(result.stdout) != _normalize_output(test["output"]):
                return ScoreResult(False, f"wrong output on test {i}")
        return ScoreResult(True, "passed")


def _extract_last_number(text: str) -> str | None:
    """Last number in the text (handles 'The answer is 42.' and thousands commas)."""
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return nums[-1].rstrip(".") if nums else None


class GSM8K:
    """GSM8K grade-school math — deterministic numeric exact-match. A reasoning
    task type distinct from code, for the cross-task comparison."""

    name = "gsm8k"

    _SYSTEM = (
        "You are a math expert. Solve the problem with brief step-by-step reasoning, "
        "then end with a line of exactly: 'The answer is <number>.'"
    )

    def load(self, limit: int | None = None) -> list[Task]:
        from datasets import load_dataset

        rows = load_dataset("openai/gsm8k", "main")["test"]
        if limit is not None:
            rows = rows.select(range(min(limit, len(rows))))
        tasks: list[Task] = []
        for i, row in enumerate(rows):
            gold = row["answer"].split("####")[-1].strip().replace(",", "")
            tasks.append(
                Task(
                    task_id=f"gsm8k/{i}",
                    messages=[
                        {"role": "system", "content": self._SYSTEM},
                        {"role": "user", "content": row["question"]},
                    ],
                    meta={"gold": gold},
                )
            )
        return tasks

    def score(self, task: Task, output: str) -> ScoreResult:
        pred = _extract_last_number(output)
        if pred is None:
            return ScoreResult(False, "failed: no number in output")
        try:
            ok = float(pred) == float(task.meta["gold"])
        except ValueError:
            ok = False
        return ScoreResult(ok, "passed" if ok else f"failed: got {pred}, want {task.meta['gold']}")


BENCHMARKS = {
    HumanEval.name: HumanEval,
    LiveCodeBench.name: LiveCodeBench,
    GSM8K.name: GSM8K,
}


def get_benchmark(name: str):
    try:
        return BENCHMARKS[name]()
    except KeyError:
        known = ", ".join(sorted(BENCHMARKS))
        raise KeyError(f"unknown benchmark '{name}'. Available: {known}") from None
