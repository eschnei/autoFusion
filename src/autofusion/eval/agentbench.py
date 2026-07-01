"""A local, Docker-free bug-fix suite for the agentic eval (Phase C).

Each task is a self-contained mini-repo: a source file with a real bug and a
test file that fails until it's fixed. No network, no Docker — the agent reads
the repo, finds the bug, edits the source, and the test command is the verifier.

`fix` holds the reference solution. It is NEVER materialized (the agent never
sees it) — it exists so the suite can self-validate: buggy == fail, fixed ==
pass. A task whose fix doesn't flip the test is a broken thermometer, and
`test_agenteval.py` asserts every task passes that check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PROMPT = (
    "This repository has a failing test caused by a bug in the source code. "
    "Run the test to see the failure, find the bug in the source, and fix it so "
    "the test passes. Do not modify the test file."
)


@dataclass
class AgentTask:
    id: str
    files: dict[str, str]      # materialized into the repo (buggy source + test)
    test_cmd: str              # exit 0 == fixed
    fix: dict[str, str] = field(default_factory=dict)  # reference solution (test-only)
    prompt: str = DEFAULT_PROMPT

    def materialize(self, dest: str | Path) -> None:
        for rel, content in self.files.items():
            p = Path(dest) / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)


# --------------------------------------------------------------------------- #
# The suite — varied difficulty so cascade/best-of differences can surface.
# Tests use plain asserts + `python <test>.py` (no pytest dep in the temp repo).
# --------------------------------------------------------------------------- #

LOCAL_SUITE: list[AgentTask] = [
    AgentTask(
        id="arithmetic",  # easy: swapped operators
        files={
            "mathx.py": "def add(a, b):\n    return a - b\n\n\ndef mul(a, b):\n    return a + b\n",
            "test_mathx.py": (
                "from mathx import add, mul\n"
                "assert add(2, 3) == 5\n"
                "assert mul(4, 5) == 20\n"
                "print('ok')\n"
            ),
        },
        test_cmd="python test_mathx.py",
        fix={"mathx.py": "def add(a, b):\n    return a + b\n\n\ndef mul(a, b):\n    return a * b\n"},
    ),
    AgentTask(
        id="fizzbuzz",  # easy-medium: wrong string on the 3-and-5 branch
        files={
            "fizzbuzz.py": (
                "def fizzbuzz(n):\n"
                "    out = []\n"
                "    for i in range(1, n + 1):\n"
                "        if i % 3 == 0 and i % 5 == 0:\n"
                "            out.append('fizz')\n"
                "        elif i % 3 == 0:\n"
                "            out.append('fizz')\n"
                "        elif i % 5 == 0:\n"
                "            out.append('buzz')\n"
                "        else:\n"
                "            out.append(str(i))\n"
                "    return out\n"
            ),
            "test_fizzbuzz.py": (
                "from fizzbuzz import fizzbuzz\n"
                "assert fizzbuzz(15)[-1] == 'fizzbuzz'\n"
                "assert fizzbuzz(5) == ['1', '2', 'fizz', '4', 'buzz']\n"
                "print('ok')\n"
            ),
        },
        test_cmd="python test_fizzbuzz.py",
        fix={
            "fizzbuzz.py": (
                "def fizzbuzz(n):\n"
                "    out = []\n"
                "    for i in range(1, n + 1):\n"
                "        if i % 3 == 0 and i % 5 == 0:\n"
                "            out.append('fizzbuzz')\n"
                "        elif i % 3 == 0:\n"
                "            out.append('fizz')\n"
                "        elif i % 5 == 0:\n"
                "            out.append('buzz')\n"
                "        else:\n"
                "            out.append(str(i))\n"
                "    return out\n"
            )
        },
    ),
    AgentTask(
        id="sliding_window",  # medium: off-by-one in the range bound
        files={
            "windows.py": (
                "def sliding_max(nums, k):\n"
                '    """Max of each contiguous window of size k."""\n'
                "    out = []\n"
                "    for i in range(len(nums) - k):\n"
                "        out.append(max(nums[i:i + k]))\n"
                "    return out\n"
            ),
            "test_windows.py": (
                "from windows import sliding_max\n"
                "assert sliding_max([1, 3, 2, 5, 4], 2) == [3, 3, 5, 5]\n"
                "assert sliding_max([4], 1) == [4]\n"
                "print('ok')\n"
            ),
        },
        test_cmd="python test_windows.py",
        fix={
            "windows.py": (
                "def sliding_max(nums, k):\n"
                '    """Max of each contiguous window of size k."""\n'
                "    out = []\n"
                "    for i in range(len(nums) - k + 1):\n"
                "        out.append(max(nums[i:i + k]))\n"
                "    return out\n"
            )
        },
    ),
    AgentTask(
        id="stack_pop",  # medium: pop reads the top but never removes it
        files={
            "stack.py": (
                "class Stack:\n"
                "    def __init__(self):\n"
                "        self._items = []\n"
                "\n"
                "    def push(self, x):\n"
                "        self._items.append(x)\n"
                "\n"
                "    def pop(self):\n"
                "        return self._items[-1]\n"
                "\n"
                "    def peek(self):\n"
                "        return self._items[-1]\n"
                "\n"
                "    def __len__(self):\n"
                "        return len(self._items)\n"
            ),
            "test_stack.py": (
                "from stack import Stack\n"
                "s = Stack()\n"
                "s.push(1)\n"
                "s.push(2)\n"
                "assert s.pop() == 2\n"
                "assert len(s) == 1\n"
                "assert s.pop() == 1\n"
                "assert len(s) == 0\n"
                "print('ok')\n"
            ),
        },
        test_cmd="python test_stack.py",
        fix={
            "stack.py": (
                "class Stack:\n"
                "    def __init__(self):\n"
                "        self._items = []\n"
                "\n"
                "    def push(self, x):\n"
                "        self._items.append(x)\n"
                "\n"
                "    def pop(self):\n"
                "        return self._items.pop()\n"
                "\n"
                "    def peek(self):\n"
                "        return self._items[-1]\n"
                "\n"
                "    def __len__(self):\n"
                "        return len(self._items)\n"
            )
        },
    ),
    AgentTask(
        id="merge_intervals",  # harder: subtle — overwrites end instead of max()
        files={
            "intervals.py": (
                "def merge_intervals(intervals):\n"
                '    """Merge overlapping [start, end] intervals."""\n'
                "    merged = []\n"
                "    for s, e in sorted(intervals):\n"
                "        if merged and s <= merged[-1][1]:\n"
                "            merged[-1][1] = e\n"
                "        else:\n"
                "            merged.append([s, e])\n"
                "    return merged\n"
            ),
            "test_intervals.py": (
                "from intervals import merge_intervals\n"
                "assert merge_intervals([[1, 4], [2, 3]]) == [[1, 4]]\n"
                "assert merge_intervals([[1, 3], [2, 6], [8, 10]]) == [[1, 6], [8, 10]]\n"
                "print('ok')\n"
            ),
        },
        test_cmd="python test_intervals.py",
        fix={
            "intervals.py": (
                "def merge_intervals(intervals):\n"
                '    """Merge overlapping [start, end] intervals."""\n'
                "    merged = []\n"
                "    for s, e in sorted(intervals):\n"
                "        if merged and s <= merged[-1][1]:\n"
                "            merged[-1][1] = max(merged[-1][1], e)\n"
                "        else:\n"
                "            merged.append([s, e])\n"
                "    return merged\n"
            )
        },
    ),
]

_SUITES = {"local": LOCAL_SUITE}


def get_suite(name: str = "local") -> list[AgentTask]:
    if name not in _SUITES:
        raise ValueError(f"unknown suite '{name}' (have: {', '.join(_SUITES)})")
    return _SUITES[name]
