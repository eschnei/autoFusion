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


def _decode_private_tests(raw: str) -> list:
    """LiveCodeBench compresses private tests as base64(zlib(pickle(...)))."""
    import base64
    import pickle
    import zlib

    try:
        return json.loads(raw)  # some rows are plain JSON
    except Exception:
        # Trusted, widely-used benchmark dataset; pickle is how it ships.
        dec = pickle.loads(zlib.decompress(base64.b64decode(raw.encode("utf-8"))))
        return json.loads(dec) if isinstance(dec, str) else dec


class LiveCodeBench:
    """LiveCodeBench (code_generation_lite) — contamination-resistant competitive
    programming. Frontier models score ~50-70%, so there's real headroom.

    FAIR grading: score() runs the **private/held-out** tests, while
    `held_out_verifier` exposes only the **public** tests — so a verify-and-select
    strategy (bestofMarj) picks on public tests but is graded on hidden ones,
    exactly like the real benchmark. stdin/stdout problems only (functional
    problems skipped + logged).
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
                public = [t for t in json.loads(rec["public_test_cases"])
                          if t.get("testtype") == "stdin"]
                if not public:  # functional/LeetCode — out of scope
                    skipped += 1
                    continue
                private = [t for t in _decode_private_tests(rec["private_test_cases"])
                           if t.get("testtype") == "stdin"]
                tasks.append(
                    Task(
                        task_id=rec["question_id"],
                        messages=[
                            {"role": "system", "content": self._SYSTEM},
                            {"role": "user", "content": rec["question_content"]},
                        ],
                        meta={"public": public, "private": private,
                              "difficulty": rec.get("difficulty")},
                    )
                )
                if limit is not None and len(tasks) >= limit:
                    break
        if skipped:
            print(f"[livecodebench] skipped {skipped} non-stdin (functional) problems")
        return tasks

    @staticmethod
    def _run_suite(code: str, tests: list) -> tuple[bool, str]:
        for i, test in enumerate(tests):
            result = run_with_stdin(code, test["input"], timeout=10.0)
            if result.timed_out:
                return False, f"timeout on test {i}"
            if _normalize_output(result.stdout) != _normalize_output(test["output"]):
                return False, f"wrong output on test {i}"
        return True, "passed"

    def score(self, task: Task, output: str) -> ScoreResult:
        # Grade on the HELD-OUT private tests (fall back to public if none shipped).
        tests = task.meta["private"] or task.meta["public"]
        passed, detail = self._run_suite(extract_code(output), tests)
        return ScoreResult(passed, detail)

    def held_out_verifier(self, task: Task):
        """The selection verifier a best-of-N strategy may use: PUBLIC tests only."""
        public = task.meta["public"]

        def verify(model_output: str) -> bool:
            return self._run_suite(extract_code(model_output), public)[0]

        return verify


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


def _extract_choice_letter(text: str) -> str | None:
    """Pull the answer letter (A–J) from a model reply — prefers an explicit
    'answer is X', falls back to the last standalone letter."""
    m = re.findall(r"answer\W*(?:is|:)?\s*\(?([A-J])\)?", text, re.IGNORECASE)
    if m:
        return m[-1].upper()
    m = re.findall(r"\b([A-J])\b", text)
    return m[-1].upper() if m else None


class MMLUPro:
    """MMLU-Pro — hard, 10-option multiple-choice across disciplines. The reasoning
    axis: single-turn, deterministic letter grading, not saturated (frontier
    ~70-85%). No test-verifier, so a best-of-N strategy falls back to its critic."""

    name = "mmlu-pro"

    _SYSTEM = (
        "You are an expert answering a hard multiple-choice question. Reason briefly, "
        "then end with a line of exactly: 'The answer is X.' where X is the option letter."
    )

    def load(self, limit: int | None = None) -> list[Task]:
        from datasets import load_dataset

        rows = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        if limit is not None:
            rows = rows.select(range(min(limit, len(rows))))
        tasks: list[Task] = []
        for i, row in enumerate(rows):
            letters = [chr(65 + j) for j in range(len(row["options"]))]
            body = "\n".join(f"{L}) {o}" for L, o in zip(letters, row["options"]))
            tasks.append(
                Task(
                    task_id=f"mmlu-pro/{row.get('question_id', i)}",
                    messages=[
                        {"role": "system", "content": self._SYSTEM},
                        {"role": "user", "content": f"{row['question']}\n\n{body}"},
                    ],
                    meta={"gold": row["answer"]},  # the correct option letter
                )
            )
        return tasks

    def score(self, task: Task, output: str) -> ScoreResult:
        pred = _extract_choice_letter(output)
        if pred is None:
            return ScoreResult(False, "failed: no letter in output")
        ok = pred == task.meta["gold"]
        return ScoreResult(ok, "passed" if ok else f"failed: got {pred}, want {task.meta['gold']}")


BENCHMARKS = {
    HumanEval.name: HumanEval,
    LiveCodeBench.name: LiveCodeBench,
    GSM8K.name: GSM8K,
    MMLUPro.name: MMLUPro,
}


def get_benchmark(name: str):
    try:
        return BENCHMARKS[name]()
    except KeyError:
        known = ", ".join(sorted(BENCHMARKS))
        raise KeyError(f"unknown benchmark '{name}'. Available: {known}") from None
