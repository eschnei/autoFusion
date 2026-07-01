"""Generate slide-ready charts from autoFusion's real benchmark results.

Three outputs in presentation/:
  1. livecodebench_bars.png   — accuracy bar chart (our recipes highlighted)
  2. quality_vs_cost.png      — the money slide: quality vs $/task (log)
  3. leaderboard.png          — dark leaderboard table across 3 benchmarks

Data are the ACTUAL measured results (LiveCodeBench n=10; HumanEval/GSM8K n=15).
Run: uv run python scripts/present_charts.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa: F401 (ensures fonts load)

OUT = Path("presentation")
OUT.mkdir(exist_ok=True)

MARJ = "#2A9D8F"      # our recipes — vivid teal
MARJ_DK = "#1F7A6F"
BASE = "#9AA5B1"      # base models — muted gray
BASE_DK = "#6B7480"
INK = "#1D2733"

# name, provider, humaneval%, gsm8k%, livecodebench%, $/task(LCB), is_marj
ROWS = [
    ("bestofMarj",  "autoFusion",  100, 100, 100, 0.00100, True),
    ("fusionMarj",  "autoFusion",  100,  93, 100, 0.03116, True),
    ("Opus 4.8",    "Anthropic",   100, 100,  90, 0.00933, False),
    ("cascadeMarj", "autoFusion",  100, 100,  70, 0.01141, True),
    ("DeepSeek-V3", "OpenRouter",  100, 100,  60, 0.00033, False),
    ("GPT-4o",      "OpenAI",      100,  87,  40, 0.00416, False),
]

SUB = "Early results · LiveCodeBench (stdin subset, public tests) · n=10 · pass@1"


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#CBD2D9",
        "axes.linewidth": 1.0,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
    })


def chart_bars():
    rows = sorted(ROWS, key=lambda r: r[4], reverse=True)
    names = [r[0] for r in rows]
    scores = [r[4] for r in rows]
    colors = [MARJ if r[6] else BASE for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160)
    fig.patch.set_facecolor("white")
    bars = ax.bar(names, scores, color=colors, width=0.66, zorder=3)
    for b, s in zip(bars, scores):
        ax.text(b.get_x() + b.get_width() / 2, s + 1.5, f"{s:.0f}",
                ha="center", va="bottom", fontsize=13, fontweight="bold", color=INK)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Accuracy  (pass@1, %)", fontsize=12)
    ax.set_title("Hard coding — LiveCodeBench", fontsize=17, fontweight="bold", pad=18)
    ax.text(0.5, 1.015, SUB, transform=ax.transAxes, ha="center", fontsize=9.5, color=BASE_DK)
    ax.grid(axis="y", color="#EAEEF2", zorder=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
    ax.set_xticklabels(names, fontsize=11)
    # legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=MARJ, label="autoFusion recipe (Marj)"),
                       Patch(color=BASE, label="single model")],
              loc="lower left", frameon=False, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(OUT / "livecodebench_bars.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def chart_quality_vs_cost():
    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=160)
    fig.patch.set_facecolor("white")
    for name, _prov, _he, _gsm, lcb, cost, marj in ROWS:
        ax.scatter(cost, lcb, s=260 if marj else 150, zorder=3,
                   color=MARJ if marj else BASE,
                   edgecolor=MARJ_DK if marj else BASE_DK, linewidth=1.5)
        dy = 2.6 if name != "cascadeMarj" else -4.2
        ax.annotate(name, (cost, lcb), textcoords="offset points",
                    xytext=(0, 10 if marj else 8), ha="center",
                    fontsize=10.5, fontweight="bold" if marj else "normal",
                    color=MARJ_DK if marj else INK)
    ax.set_xscale("log")
    ax.set_xlabel("←  cheaper     Cost per task  ($, log scale)     pricier  →", fontsize=12)
    ax.set_ylabel("Accuracy  (pass@1, %)", fontsize=12)
    ax.set_ylim(30, 108)
    ax.set_title("Quality vs. cost per task — LiveCodeBench", fontsize=17, fontweight="bold", pad=18)
    ax.text(0.5, 1.015, SUB, transform=ax.transAxes, ha="center", fontsize=9.5, color=BASE_DK)
    ax.grid(color="#EAEEF2", zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    # "best" corner call-out
    ax.annotate("best:  high quality, low cost",
                xy=(0.0011, 100), xytext=(0.0016, 84),
                fontsize=10, color=MARJ_DK, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=MARJ_DK, lw=1.6))
    fig.tight_layout()
    fig.savefig(OUT / "quality_vs_cost.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def chart_leaderboard():
    bg, panel, line = "#0F1620", "#1A2430", "#2A3644"
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=160)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.axis("off")

    cols = ["#", "Model", "Provider", "HumanEval", "GSM8K", "LiveCodeBench", "$/task"]
    xs = [0.02, 0.09, 0.30, 0.47, 0.60, 0.735, 0.90]
    rows = sorted(ROWS, key=lambda r: r[4], reverse=True)

    ax.text(0.01, 1.05, "autoFusion — recipes vs. models",
            color="white", fontsize=15, fontweight="bold", transform=ax.transAxes)

    y0, dy = 0.83, 0.125
    # header
    for x, c in zip(xs, cols):
        ax.text(x, y0 + 0.08, c, color="#8A97A6", fontsize=11, fontweight="bold",
                transform=ax.transAxes)
    ax.plot([0.01, 0.99], [y0 + 0.05, y0 + 0.05], color=line, lw=1, transform=ax.transAxes)

    for i, (name, prov, he, gsm, lcb, cost, marj) in enumerate(rows):
        y = y0 - i * dy
        if marj:  # tinted row band for our recipes
            ax.add_patch(plt.Rectangle((0.01, y - 0.038), 0.98, 0.10, transform=ax.transAxes,
                                       color=panel, zorder=0))
        nm_color = MARJ if marj else "white"
        ax.text(xs[0], y, str(i + 1), color="#6B7480", fontsize=12, transform=ax.transAxes)
        ax.text(xs[1], y, name, color=nm_color, fontsize=12.5, fontweight="bold", transform=ax.transAxes)
        ax.text(xs[2], y, prov, color="#B7C0CC", fontsize=11, transform=ax.transAxes)
        ax.text(xs[3], y, f"{he:.0f}", color="#7FD1C4", fontsize=12, transform=ax.transAxes)
        ax.text(xs[4], y, f"{gsm:.0f}", color="#7FD1C4", fontsize=12, transform=ax.transAxes)
        # LiveCodeBench = the differentiator: color by value
        lcb_c = "#39D98A" if lcb >= 90 else ("#E8C468" if lcb >= 60 else "#E86A5E")
        ax.text(xs[5], y, f"{lcb:.0f}", color=lcb_c, fontsize=13, fontweight="bold", transform=ax.transAxes)
        ax.text(xs[6], y, f"${cost:.4f}", color="#B7C0CC", fontsize=11.5, transform=ax.transAxes)

    ax.text(0.01, -0.02,
            "Scores = pass@1 %. HumanEval/GSM8K n=15 (saturated). LiveCodeBench n=10 (public tests). "
            "Marj rows = autoFusion recipes over cheaper models.",
            color="#6B7480", fontsize=8.5, transform=ax.transAxes)
    fig.savefig(OUT / "leaderboard.png", bbox_inches="tight", facecolor=bg)
    plt.close(fig)


def main():
    _style()
    chart_bars()
    chart_quality_vs_cost()
    chart_leaderboard()
    print("wrote:", ", ".join(str(p) for p in sorted(OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
