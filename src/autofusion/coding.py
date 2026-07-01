"""Verify-in-the-loop coding (Phase 11) — the proven bestofMarj win, on real code.

Given a bounded task + a way to run tests, sample N candidate solutions from a
coding basket, write each to the target file, run the user's test command, and
keep the first that passes. This is `VerifiedBestOfN` with a REAL verifier (the
user's tests) instead of a benchmark scorer.

SECURITY: this writes model-generated code to `target_file` and runs `test_cmd`
in a subprocess. Only run it on a task you'd run those tests for yourself.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .eval.benchmarks import extract_code

CODE_SYSTEM = (
    "You are an expert programmer. Implement the task as a COMPLETE, self-contained "
    "solution written to the requested file. Return ONLY the code in a single code "
    "block — no explanation, no prose."
)


def build_task_messages(task: str, target_file: str, context_files: list[str]) -> list[dict]:
    ctx = ""
    for f in context_files or []:
        try:
            ctx += f"\n\n# ---- {f} ----\n{Path(f).read_text()}"
        except OSError as exc:
            ctx += f"\n\n# ---- {f} (unreadable: {exc}) ----"
    user = task
    if ctx:
        user += f"\n\nRelevant files:{ctx}"
    user += f"\n\nWrite the full solution to `{target_file}`."
    return [{"role": "system", "content": CODE_SYSTEM}, {"role": "user", "content": user}]


def build_verifier(target_file: str, test_cmd: str, timeout: float = 60.0):
    """Verifier for VerifiedBestOfN: write the candidate to `target_file`, run
    `test_cmd`, pass iff it exits 0. Returns None if there's no test command
    (bestofMarj then falls back to its critic)."""
    if not test_cmd:
        return None

    def verify(model_output: str) -> bool:
        code = extract_code(model_output)
        Path(target_file).write_text(code)
        try:
            proc = subprocess.run(
                test_cmd, shell=True, capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            return False
        return proc.returncode == 0

    return verify
