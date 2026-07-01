"""SWE-bench slice (Phase D) — real GitHub issues, agent-generated patches.

Two stages, deliberately decoupled so the heavy one is opt-in:

  1. PREDICT (Docker-free, runnable here): for each instance, checkout the repo at
     `base_commit`, run a single-agent trajectory to edit the code toward the issue,
     then `git diff` -> a `model_patch`. Writes predictions in the official
     SWE-bench format ({instance_id, model_name_or_path, model_patch}).

  2. GRADE (Docker): hand the predictions to the official `swebench` harness, which
     applies each patch + the instance's hidden `test_patch` inside the instance's
     environment image and checks FAIL_TO_PASS / PASS_TO_PASS. We parse its report.

The agent edits WITHOUT running the repo's real tests — that environment lives in
the Docker image, not here — so PREDICT measures issue-solving from the problem
statement + code reading, and the official harness is the honest grader. (Verify-
in-the-loop on real repos is the Docker-hosted next step; this keeps the two
concerns cleanly separable and the grader trusted.)
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..agent import Workspace, run_agent
from ..budget import BudgetTracker

DEFAULT_DATASET = "princeton-nlp/SWE-bench_Lite"
MODEL_PREFIX = "autofusion"

SWE_PROMPT = (
    "Resolve the following GitHub issue by editing the repository's source code.\n\n"
    "IMPORTANT: the test environment's dependencies are NOT installed here, so the "
    "repo's test suite will not run — do not rely on running the tests. Instead read "
    "the relevant source, reason carefully about the root cause from the issue, and "
    "make a focused, correct edit. Use read_file/grep to locate the code. When your "
    "edit is complete, call finish.\n\n"
    "--- ISSUE ---\n{problem}\n"
)


@dataclass
class SweInstance:
    instance_id: str
    repo: str          # "owner/name"
    base_commit: str
    problem_statement: str

    @classmethod
    def from_row(cls, row: dict) -> "SweInstance":
        return cls(
            instance_id=row["instance_id"],
            repo=row["repo"],
            base_commit=row["base_commit"],
            problem_statement=row["problem_statement"],
        )


def load_instances(
    dataset: str = DEFAULT_DATASET, split: str = "test",
    limit: int | None = None, ids: list[str] | None = None,
) -> list[SweInstance]:
    from datasets import load_dataset

    ds = load_dataset(dataset, split=split)
    want = set(ids) if ids else None
    out: list[SweInstance] = []
    for row in ds:
        if want is not None and row["instance_id"] not in want:
            continue
        out.append(SweInstance.from_row(row))
        if limit and want is None and len(out) >= limit:
            break
    return out


def _run(cmd: list[str], cwd: str | None = None, timeout: float = 600.0) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def checkout_repo(repo: str, base_commit: str, dest: str | Path) -> Path:
    """Blob-filtered clone of `repo` checked out at `base_commit` (fast, small)."""
    dest = Path(dest)
    url = f"https://github.com/{repo}.git"
    _run(["git", "clone", "--filter=blob:none", "--quiet", url, str(dest)])
    _run(["git", "checkout", "--quiet", base_commit], cwd=str(dest))
    return dest


def diff_patch(repo_dir: str | Path) -> str:
    """The agent's edits as a unified diff — the `model_patch` the grader applies."""
    return _run(["git", "diff"], cwd=str(repo_dir)).stdout


@dataclass
class Prediction:
    instance_id: str
    model_name_or_path: str
    model_patch: str
    cost_usd: float = 0.0
    n_calls: int = 0

    def to_record(self) -> dict:  # official SWE-bench prediction shape
        return {
            "instance_id": self.instance_id,
            "model_name_or_path": self.model_name_or_path,
            "model_patch": self.model_patch,
        }


async def predict_instance(
    spec, inst: SweInstance, *, budget: BudgetTracker | None = None, max_steps: int = 30,
    label: str | None = None,
) -> Prediction:
    tmp = Path(tempfile.mkdtemp(prefix="af-swe-"))
    repo_dir = tmp / "repo"
    try:
        checkout_repo(inst.repo, inst.base_commit, repo_dir)
        ws = Workspace(repo_dir)
        prompt = SWE_PROMPT.format(problem=inst.problem_statement[:6000])
        res = await run_agent(spec, prompt, ws, budget=budget, max_steps=max_steps)
        patch = diff_patch(repo_dir)
        return Prediction(
            instance_id=inst.instance_id,
            model_name_or_path=f"{MODEL_PREFIX}.{label or spec.name}",
            model_patch=patch, cost_usd=res.cost_usd, n_calls=res.n_calls,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


async def predict(
    spec, instances: list[SweInstance], *, budget: BudgetTracker | None = None,
    max_steps: int = 30, concurrency: int = 3, label: str | None = None,
) -> list[Prediction]:
    sem = asyncio.Semaphore(concurrency)

    async def one(inst: SweInstance) -> Prediction:
        async with sem:
            return await predict_instance(spec, inst, budget=budget, max_steps=max_steps, label=label)

    return await asyncio.gather(*[one(i) for i in instances])


def write_predictions(preds: list[Prediction], path: str | Path) -> Path:
    path = Path(path)
    with path.open("w") as fh:
        for p in preds:
            fh.write(json.dumps(p.to_record()) + "\n")
    return path


def grade(
    predictions_path: str | Path, run_id: str, *, dataset: str = DEFAULT_DATASET,
    ids: list[str] | None = None, max_workers: int = 2, timeout: float = 5400.0,
) -> subprocess.CompletedProcess:
    """Invoke the official swebench Docker harness. Requires `pip install swebench` +
    a running Docker daemon. Writes a `<model>.<run_id>.json` report in the cwd."""
    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", dataset, "--predictions_path", str(predictions_path),
        "--run_id", run_id, "--max_workers", str(max_workers),
    ]
    if ids:
        cmd += ["--instance_ids", *ids]
    return _run(cmd, timeout=timeout)


def parse_report(report_path: str | Path) -> dict:
    """Read the harness's report JSON into a compact summary."""
    data = json.loads(Path(report_path).read_text())
    return {
        "total": data.get("total_instances", 0),
        "submitted": data.get("submitted_instances", 0),
        "resolved": data.get("resolved_instances", 0),
        "unresolved": data.get("unresolved_instances", 0),
        "errors": data.get("error_instances", 0),
        "resolved_ids": data.get("resolved_ids", []),
    }
