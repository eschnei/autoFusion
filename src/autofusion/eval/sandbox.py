"""Sandboxed execution of model-generated code (Phase 1).

SECURITY: HumanEval grading runs untrusted, model-written code. This module
isolates each program in a fresh subprocess with a hard wall-clock timeout,
CPU/memory/file-size resource limits, and a reliability guard that neuters the
most destructive syscalls. This is the human-eval-style guard — adequate for
benchmark models you control, NOT a security boundary for adversarial code.
For untrusted-at-scale use, run inside a locked-down container (gVisor/seccomp,
no network). See SECURITY note in the README.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass

# Prepended to every program before execution. Disables destructive operations
# and caps resources from inside the child (belt-and-suspenders with the
# subprocess timeout). Wrapped in try/except since some RLIMITs aren't
# enforced on macOS.
_RELIABILITY_GUARD = '''\
import faulthandler, builtins, os, shutil, subprocess as _sp
faulthandler.disable()
try:
    import resource
    _cpu = 8
    resource.setrlimit(resource.RLIMIT_CPU, (_cpu, _cpu))
    _mem = 512 * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (_mem, _mem))
    except (ValueError, OSError):
        pass
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
except Exception:
    pass
os.system = None
os.remove = os.unlink = os.rmdir = os.removedirs = None
shutil.rmtree = None
_sp.Popen = None
import sys as _sys
_sys.setrecursionlimit(100000)
'''


@dataclass
class ExecResult:
    passed: bool
    detail: str  # "passed" | "failed: ..." | "timeout" | "error: ..."


@dataclass
class RunOutput:
    returncode: int
    stdout: str
    timed_out: bool = False


def run_python(program: str, timeout: float = 10.0) -> ExecResult:
    """Run a self-contained Python program; pass == clean exit (returncode 0)."""
    full = _RELIABILITY_GUARD + "\n" + program
    with tempfile.TemporaryDirectory() as workdir:
        prog_path = os.path.join(workdir, "prog.py")
        with open(prog_path, "w") as fh:
            fh.write(full)
        # Minimal env: no inherited proxy/network hints, isolated cwd.
        env = {"PATH": os.environ.get("PATH", ""), "HOME": workdir, "TMPDIR": workdir}
        try:
            proc = subprocess.run(
                [sys.executable, "-I", prog_path],  # -I: isolated, ignore env/site
                cwd=workdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(False, "timeout")
        if proc.returncode == 0:
            return ExecResult(True, "passed")
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return ExecResult(False, "failed: " + (tail[-1] if tail else f"exit {proc.returncode}"))


def run_with_stdin(program: str, stdin: str = "", timeout: float = 10.0) -> RunOutput:
    """Run a program feeding `stdin`, capturing stdout (for stdin/stdout benchmarks
    like LiveCodeBench). Same isolation + reliability guard as run_python."""
    full = _RELIABILITY_GUARD + "\n" + program
    with tempfile.TemporaryDirectory() as workdir:
        prog_path = os.path.join(workdir, "prog.py")
        with open(prog_path, "w") as fh:
            fh.write(full)
        env = {"PATH": os.environ.get("PATH", ""), "HOME": workdir, "TMPDIR": workdir}
        try:
            proc = subprocess.run(
                [sys.executable, "-I", prog_path],
                cwd=workdir, env=env, input=stdin,
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return RunOutput(returncode=-1, stdout="", timed_out=True)
        return RunOutput(returncode=proc.returncode, stdout=proc.stdout)
