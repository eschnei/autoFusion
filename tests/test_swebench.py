"""SWE-bench harness (Phase D) — the Docker-free, model-free logic:
patch extraction, official prediction format, and report parsing."""

import asyncio
import json
import subprocess

from autofusion.config import ModelSpec
from autofusion.eval import swebench as swe


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_diff_patch_captures_agent_edits(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t.co")
    _git(repo, "config", "user.name", "t")
    (repo / "mod.py").write_text("def f():\n    return 0\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")

    (repo / "mod.py").write_text("def f():\n    return 1\n")   # "agent" edit
    patch = swe.diff_patch(repo)
    assert patch.startswith("diff --git")
    assert "-    return 0" in patch and "+    return 1" in patch


def test_empty_diff_when_no_edit(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t.co")
    _git(repo, "config", "user.name", "t")
    (repo / "a.txt").write_text("x\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    assert swe.diff_patch(repo).strip() == ""       # no edit -> empty patch


def test_prediction_record_is_official_shape():
    p = swe.Prediction(instance_id="astropy__astropy-1",
                       model_name_or_path="autofusion.gpt-4o",
                       model_patch="diff --git ...", cost_usd=0.9, n_calls=5)
    rec = p.to_record()
    assert set(rec) == {"instance_id", "model_name_or_path", "model_patch"}   # exactly the 3 fields
    assert rec["model_patch"] == "diff --git ..."


def test_write_predictions_is_jsonl(tmp_path):
    preds = [
        swe.Prediction("i1", "autofusion.m", "patchA"),
        swe.Prediction("i2", "autofusion.m", "patchB"),
    ]
    path = swe.write_predictions(preds, tmp_path / "preds.jsonl")
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["instance_id"] == "i1"
    assert "cost_usd" not in json.loads(lines[0])       # internal fields stay out of the file


def test_instance_from_row():
    row = {"instance_id": "django__django-1", "repo": "django/django",
           "base_commit": "abc123", "problem_statement": "boom", "extra": "ignored"}
    inst = swe.SweInstance.from_row(row)
    assert inst.repo == "django/django" and inst.base_commit == "abc123"


def test_parse_report(tmp_path):
    report = tmp_path / "autofusion.gpt-4o.run1.json"
    report.write_text(json.dumps({
        "total_instances": 3, "submitted_instances": 2, "resolved_instances": 1,
        "unresolved_instances": 1, "error_instances": 0, "resolved_ids": ["django__django-1"],
    }))
    s = swe.parse_report(report)
    assert s["resolved"] == 1 and s["submitted"] == 2
    assert s["resolved_ids"] == ["django__django-1"]


def test_predict_uses_mocked_agent(tmp_path, monkeypatch):
    """predict_instance: checkout is stubbed, agent is stubbed -> we exercise the
    orchestration (prompt, diff, prediction assembly) without network/model."""
    inst = swe.SweInstance("proj__proj-1", "proj/proj", "deadbeef", "fix the bug")

    def fake_checkout(repo, base_commit, dest):
        from pathlib import Path
        d = Path(dest)
        d.mkdir(parents=True, exist_ok=True)
        (d / "src.py").write_text("buggy\n")
        return d

    async def fake_agent(spec, task, ws, **kw):
        assert "fix the bug" in task              # the issue reached the agent
        ws.write_file("src.py", "fixed\n")
        return __import__("autofusion.agent", fromlist=["AgentResult"]).AgentResult(
            "done", 0.02, 3, 3, True)

    monkeypatch.setattr(swe, "checkout_repo", fake_checkout)
    monkeypatch.setattr(swe, "diff_patch", lambda d: "diff --git a/src.py b/src.py\n+fixed")
    monkeypatch.setattr(swe, "run_agent", fake_agent)

    spec = ModelSpec(name="gpt-4o", model="gpt-4o")
    pred = asyncio.run(swe.predict_instance(spec, inst))
    assert pred.instance_id == "proj__proj-1"
    assert pred.model_name_or_path == "autofusion.gpt-4o"
    assert "fixed" in pred.model_patch and pred.cost_usd == 0.02
