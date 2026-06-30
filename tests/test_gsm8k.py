"""Calibrate the GSM8K scorer (MAR-29)."""

from autofusion.eval.benchmarks import GSM8K, _extract_last_number, get_benchmark


def test_registered():
    assert isinstance(get_benchmark("gsm8k"), GSM8K)


def test_extract_last_number():
    assert _extract_last_number("First 10, then 32. The answer is 42.") == "42"
    assert _extract_last_number("It costs 1,200 dollars") == "1200"
    assert _extract_last_number("no digits here") is None


def test_gold_answers_score_pass():
    """The dataset's own gold answers must score PASS (the thermometer check)."""
    from datasets import load_dataset

    rows = load_dataset("openai/gsm8k", "main")["test"].select(range(10))
    bench = GSM8K()
    tasks = bench.load(limit=10)
    for task, row in zip(tasks, rows):
        # Feeding the reference answer string must grade correct.
        assert bench.score(task, row["answer"]).passed, f"{task.task_id} gold should pass"


def test_wrong_answer_fails():
    bench = GSM8K()
    task = bench.load(limit=1)[0]
    wrong = "The answer is -99999."
    assert not bench.score(task, wrong).passed
