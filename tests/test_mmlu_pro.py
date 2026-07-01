"""Calibrate the MMLU-Pro reasoning scorer (MAR-42)."""

from autofusion.eval.benchmarks import MMLUPro, _extract_choice_letter, get_benchmark


def test_registered():
    assert isinstance(get_benchmark("mmlu-pro"), MMLUPro)


def test_extract_choice_letter():
    assert _extract_choice_letter("Reasoning... The answer is D.") == "D"
    assert _extract_choice_letter("I'll go with (B)") == "B"
    assert _extract_choice_letter("clearly C") == "C"
    assert _extract_choice_letter("no letters here 123") is None


def test_gold_letters_score_pass():
    """The dataset's own gold letters must score PASS (calibration)."""
    bench = MMLUPro()
    tasks = bench.load(limit=12)
    for t in tasks:
        gold = t.meta["gold"]
        assert bench.score(t, f"The answer is {gold}.").passed, f"{t.task_id} gold should pass"


def test_wrong_letter_fails():
    bench = MMLUPro()
    t = bench.load(limit=1)[0]
    wrong = "Z" if t.meta["gold"] != "Z" else "A"
    # pick a definitely-different valid letter
    other = "A" if t.meta["gold"] != "A" else "B"
    assert not bench.score(t, f"The answer is {other}.").passed
