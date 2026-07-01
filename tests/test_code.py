"""The `code` verify-in-the-loop tool (MAR-43), model-free."""

from autofusion.coding import build_task_messages, build_verifier


def test_verifier_passes_on_correct_code(tmp_path):
    target = tmp_path / "sol.py"
    (tmp_path / "t.py").write_text("import sol; assert sol.add(2, 3) == 5\n")
    verify = build_verifier(str(target), f"cd {tmp_path} && python t.py", timeout=30)
    assert verify("```python\ndef add(a, b):\n    return a + b\n```") is True
    assert "return a + b" in target.read_text()   # candidate was written to the file


def test_verifier_fails_on_wrong_code(tmp_path):
    target = tmp_path / "sol.py"
    (tmp_path / "t.py").write_text("import sol; assert sol.add(2, 3) == 5\n")
    verify = build_verifier(str(target), f"cd {tmp_path} && python t.py", timeout=30)
    assert verify("```python\ndef add(a, b):\n    return a + b + 1\n```") is False


def test_verifier_times_out(tmp_path):
    target = tmp_path / "sol.py"
    (tmp_path / "t.py").write_text("import sol\n")   # importing sol hangs
    verify = build_verifier(str(target), f"cd {tmp_path} && python t.py", timeout=2)
    assert verify("```python\nwhile True:\n    pass\n```") is False


def test_no_test_command_returns_none():
    assert build_verifier("x.py", "") is None


def test_task_messages_include_context(tmp_path):
    ctx = tmp_path / "helpers.py"
    ctx.write_text("SECRET = 42\n")
    msgs = build_task_messages("use SECRET", "out.py", [str(ctx)])
    assert msgs[0]["role"] == "system"
    assert "SECRET = 42" in msgs[1]["content"] and "out.py" in msgs[1]["content"]
