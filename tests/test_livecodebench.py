"""Calibrate the LiveCodeBench scorer (MAR-22).

No reference solutions ship with the dataset, so we calibrate two ways:
1. A network-free plumbing check of the stdin/stdout runner.
2. A hand-authored CORRECT solution to a real problem ("Short Sort", 1873_A)
   must score PASS — proving the grading path works on real data. If it doesn't,
   the scorer is wrong and every LiveCodeBench number would be a lie.
"""

import pytest

from autofusion.eval.benchmarks import LiveCodeBench, _normalize_output, get_benchmark
from autofusion.eval.sandbox import run_with_stdin


def test_registered():
    assert isinstance(get_benchmark("livecodebench"), LiveCodeBench)


def test_normalize_output_trailing_whitespace():
    assert _normalize_output("YES \nNO\n") == _normalize_output("YES\nNO")


def test_stdin_runner_plumbing():
    prog = "n=int(input())\nprint(sum(int(input()) for _ in range(n)))"
    out = run_with_stdin(prog, stdin="3\n10\n20\n12\n", timeout=10.0)
    assert out.returncode == 0 and out.stdout.strip() == "42"


@pytest.mark.parametrize("data_file", ["test.jsonl"])
def test_known_correct_solution_scores_pass(data_file):
    bench = LiveCodeBench()
    tasks = {t.task_id: t for t in bench.load(limit=5, data_file=data_file)}
    # 1873_A "Short Sort": YES unless all 3 chars are misplaced vs "abc".
    assert "1873_A" in tasks, f"expected 1873_A in {list(tasks)}"
    solution = (
        "```python\n"
        "t = int(input())\n"
        "for _ in range(t):\n"
        "    s = input().strip()\n"
        "    print('YES' if sum(a != b for a, b in zip(s, 'abc')) != 3 else 'NO')\n"
        "```"
    )
    result = bench.score(tasks["1873_A"], solution)
    assert result.passed, f"correct solution should pass but: {result.detail}"
