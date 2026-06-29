"""Validate the eval harness against ground truth, with no model in the loop.

If HumanEval's own canonical solutions don't score as PASS, the scorer is
broken — and every model number it ever produces would be a lie. This is the
calibration check for the thermometer itself.
"""

import pytest

from autofusion.eval.benchmarks import HumanEval, extract_code
from autofusion.eval.results import pass_at_k


def test_extract_code_prefers_block_with_def():
    text = "Here you go:\n```python\ndef foo():\n    return 1\n```\nThanks!"
    assert extract_code(text) == "def foo():\n    return 1"


def test_extract_code_falls_back_to_raw():
    assert extract_code("def bar():\n    return 2") == "def bar():\n    return 2"


def test_pass_at_k_basic():
    assert pass_at_k(1, 1, 1) == 1.0
    assert pass_at_k(1, 0, 1) == 0.0
    assert round(pass_at_k(5, 1, 1), 3) == 0.2


@pytest.mark.parametrize("n", [10])
def test_canonical_solutions_score_pass(n):
    """The first n HumanEval canonical solutions must all PASS."""
    from datasets import load_dataset

    rows = load_dataset("openai/openai_humaneval")["test"].select(range(n))
    bench = HumanEval()
    tasks = bench.load(limit=n)
    by_id = {t.task_id: t for t in tasks}

    for row in rows:
        task = by_id[row["task_id"]]
        # Reconstruct the full correct function: prompt (signature+docstring) + body.
        full = "```python\n" + row["prompt"] + row["canonical_solution"] + "\n```"
        result = bench.score(task, full)
        assert result.passed, f"{row['task_id']} should pass but: {result.detail}"
