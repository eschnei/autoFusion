"""Generate slide-ready charts from autoFusion's real, FAIR benchmark results.

Outputs in presentation/:
  1. livecodebench_bars.png   — coding accuracy (held-out private grading)
  2. quality_vs_cost.png      — the money slide: quality vs $/task (log)
  3. leaderboard.png          — dark table: reasoning + coding + $/task
  4. axes_compare.png         — coding vs reasoning (where we win / where frontier wins)

Data = actual measured results:
  LiveCodeBench n=10 (graded on HELD-OUT private tests; bestofMarj selects on public).
  MMLU-Pro n=20 (reasoning; no verifier, so bestofMarj uses a critic).
Run: uv run python scripts/present_charts.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = Path("presentation")
OUT.mkdir(exist_ok=True)

MARJ, MARJ_DK = "#2A9D8F", "#1F7A6F"
BASE, BASE_DK = "#9AA5B1", "#6B7480"
INK = "#1D2733"

# name, provider, MMLU-Pro %, LiveCodeBench %, $/task (LCB), is_marj
ROWS = [
    ("bestofMarj",  "autoFusion",  75,  90, 0.00102, True),
    ("fusionMarj",  "autoFusion",  85, 100, 0.02699, True),
    ("Opus 4.8",    "Anthropic",   85, 100, 0.00826, False),
    ("cascadeMarj", "autoFusion",  65,  80, 0.01134, True),
    ("DeepSeek-V3", "OpenRouter",  70,  80, 0.00031, False),
    ("GPT-4o",      "OpenAI",      80,  40, 0.00401, False),
]
LCB_SUB = "Held-out private-test grading · n=10 · pass@1 · early results"


def _style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#CBD2D9",
                         "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK})


def chart_bars():
    rows = sorted(ROWS, key=lambda r: r[3], reverse=True)
    names, scores = [r[0] for r in rows], [r[3] for r in rows]
    colors = [MARJ if r[5] else BASE for r in rows]
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160)
    fig.patch.set_facecolor("white")
    bars = ax.bar(names, scores, color=colors, width=0.66, zorder=3)
    for b, s in zip(bars, scores):
        ax.text(b.get_x() + b.get_width() / 2, s + 1.5, f"{s:.0f}", ha="center",
                va="bottom", fontsize=13, fontweight="bold", color=INK)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Accuracy  (pass@1, %)", fontsize=12)
    ax.set_title("Hard coding — LiveCodeBench", fontsize=17, fontweight="bold", pad=18)
    ax.text(0.5, 1.015, LCB_SUB, transform=ax.transAxes, ha="center", fontsize=9.5, color=BASE_DK)
    ax.grid(axis="y", color="#EAEEF2", zorder=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(handles=[Patch(color=MARJ, label="autoFusion recipe (Marj)"),
                       Patch(color=BASE, label="single model")],
              loc="lower left", frameon=False, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(OUT / "livecodebench_bars.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def chart_quality_vs_cost():
    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=160)
    fig.patch.set_facecolor("white")
    for name, _p, _mmlu, lcb, cost, marj in ROWS:
        ax.scatter(cost, lcb, s=260 if marj else 150, zorder=3, color=MARJ if marj else BASE,
                   edgecolor=MARJ_DK if marj else BASE_DK, linewidth=1.5)
        ax.annotate(name, (cost, lcb), textcoords="offset points", xytext=(0, 11),
                    ha="center", fontsize=10.5, fontweight="bold" if marj else "normal",
                    color=MARJ_DK if marj else INK)
    ax.set_xscale("log")
    ax.set_xlabel("←  cheaper     Cost per task  ($, log scale)     pricier  →", fontsize=12)
    ax.set_ylabel("Accuracy  (pass@1, %)", fontsize=12)
    ax.set_ylim(30, 112)
    ax.set_title("Quality vs. cost per task — LiveCodeBench", fontsize=17, fontweight="bold", pad=18)
    ax.text(0.5, 1.015, LCB_SUB, transform=ax.transAxes, ha="center", fontsize=9.5, color=BASE_DK)
    ax.grid(color="#EAEEF2", zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.annotate("bestofMarj:  near-Opus quality\nat ~8× lower cost",
                xy=(0.00105, 90), xytext=(0.0022, 66), fontsize=10, color=MARJ_DK,
                fontweight="bold", arrowprops=dict(arrowstyle="->", color=MARJ_DK, lw=1.6))
    fig.tight_layout()
    fig.savefig(OUT / "quality_vs_cost.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def chart_axes_compare():
    """Grouped bars: coding vs reasoning — where recipes win, where frontier wins."""
    order = ["Opus 4.8", "fusionMarj", "bestofMarj", "GPT-4o", "DeepSeek-V3", "cascadeMarj"]
    by = {r[0]: r for r in ROWS}
    rows = [by[n] for n in order]
    import numpy as np
    x = np.arange(len(rows))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=160)
    fig.patch.set_facecolor("white")
    ax.bar(x - w / 2, [r[3] for r in rows], w, label="Coding (LiveCodeBench)", color="#2A9D8F", zorder=3)
    ax.bar(x + w / 2, [r[2] for r in rows], w, label="Reasoning (MMLU-Pro)", color="#E9A23B", zorder=3)
    ax.set_xticks(x)
    labels = [r[0] + ("  ·Marj" if r[5] else "") for r in rows]
    ax.set_xticklabels([r[0] for r in rows], fontsize=10.5)
    ax.set_ylim(0, 118)
    ax.set_ylabel("Accuracy  (pass@1, %)", fontsize=12)
    ax.set_title("Where recipes win — and where frontier still leads",
                 fontsize=16, fontweight="bold", pad=18)
    ax.text(0.5, 1.015, "Coding: verify-and-select works.  Reasoning: no verifier → frontier leads. "
            "· early results", transform=ax.transAxes, ha="center", fontsize=9.5, color=BASE_DK)
    ax.grid(axis="y", color="#EAEEF2", zorder=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "axes_compare.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def chart_leaderboard():
    bg, panel, line = "#0F1620", "#1A2430", "#2A3644"
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=160)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.axis("off")
    cols = ["#", "Model", "Provider", "Reasoning\nMMLU-Pro", "Coding\nLiveCodeBench", "$/task"]
    xs = [0.02, 0.09, 0.31, 0.53, 0.71, 0.90]
    rows = sorted(ROWS, key=lambda r: (r[3], r[2]), reverse=True)

    ax.text(0.01, 1.06, "autoFusion — recipes vs. models",
            color="white", fontsize=15, fontweight="bold", transform=ax.transAxes)
    y0, dy = 0.82, 0.125
    for x, c in zip(xs, cols):
        ax.text(x, y0 + 0.075, c, color="#8A97A6", fontsize=10.5, fontweight="bold",
                va="bottom", transform=ax.transAxes)
    ax.plot([0.01, 0.99], [y0 + 0.05, y0 + 0.05], color=line, lw=1, transform=ax.transAxes)

    for i, (name, prov, mmlu, lcb, cost, marj) in enumerate(rows):
        y = y0 - i * dy
        if marj:
            ax.add_patch(plt.Rectangle((0.01, y - 0.038), 0.98, 0.10, transform=ax.transAxes,
                                       color=panel, zorder=0))
        ax.text(xs[0], y, str(i + 1), color="#6B7480", fontsize=12, transform=ax.transAxes)
        ax.text(xs[1], y, name, color=MARJ if marj else "white", fontsize=12.5,
                fontweight="bold", transform=ax.transAxes)
        ax.text(xs[2], y, prov, color="#B7C0CC", fontsize=11, transform=ax.transAxes)
        ax.text(xs[3], y, f"{mmlu:.0f}", color="#E9C46A", fontsize=12.5, transform=ax.transAxes)
        lcb_c = "#39D98A" if lcb >= 90 else ("#E8C468" if lcb >= 60 else "#E86A5E")
        ax.text(xs[4], y, f"{lcb:.0f}", color=lcb_c, fontsize=13, fontweight="bold", transform=ax.transAxes)
        ax.text(xs[5], y, f"${cost:.4f}", color="#B7C0CC", fontsize=11.5, transform=ax.transAxes)

    ax.text(0.01, -0.03,
            "pass@1 %. MMLU-Pro n=20. LiveCodeBench n=10, graded on HELD-OUT private tests. "
            "Marj rows = autoFusion recipes over cheaper models. Early results.",
            color="#6B7480", fontsize=8.5, transform=ax.transAxes)
    fig.savefig(OUT / "leaderboard.png", bbox_inches="tight", facecolor=bg)
    plt.close(fig)


def main():
    _style()
    chart_bars()
    chart_quality_vs_cost()
    chart_axes_compare()
    chart_leaderboard()
    print("wrote:", ", ".join(p.name for p in sorted(OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
