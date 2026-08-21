#!/usr/bin/env python3
"""fig_physion_leaderboard -- the Physion-Eval flag-rate leaderboard.

A horizontal bar chart of the physical-implausibility FLAG RATE for every
evaluator in Table~\\ref{tab:physion}, sorted high to low: the untrained-human
reference on top as a light bar, this work's evidence-conditioned flag rate as
the single accented (red) bar, and the benchmark's own critics below in ink.
The other two columns of the source table (J exo / J ego) are not plotted --- a
leaderboard shows one metric --- and remain in the full table in the appendix.

Every number is transcribed verbatim from tab_physion.tex; nothing is recomputed
here. Text is set in Computer Modern to match the report body.

    python3 make_fig_physion_leaderboard.py
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

HERE = Path(__file__).resolve().parent

# ---- report visual system -------------------------------------------------- #
ADOBE_RED = "#FA0F00"   # single accent: this work's row
INK = "#1A1A1A"
GRAY = "#6E6E6E"
LIGHT = "#C9C9C9"       # the reference (human) bar
GRID = "#E8E8E8"
HAIR = 0.6

# ---- fonts: Computer Modern (matches the body CMR10) ----------------------- #
_MPL_TTF = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
CM_BOLD = os.path.join(_MPL_TTF, "cmb10.ttf")


def apply_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["cmr10", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "text.color": INK, "axes.edgecolor": GRAY,
    })


def cm_bold(size):
    return fm.FontProperties(fname=CM_BOLD, size=size)


# ---- the data (verbatim from tab_physion.tex, flag-rate column) ------------ #
# (label, flag_rate_percent, kind)  kind in {"ref","ours","critic"}
ROWS = [
    ("Untrained humans (reference)",       52.7, "ref"),
    ("This work (evidence-conditioned)",   26.8, "ours"),
    ("Gemini 3.0 Pro",                     16.9, "critic"),
    ("Qwen3-VL-32B",                       14.4, "critic"),
    ("Qwen3-VL-8B",                        12.7, "critic"),
    ("Claude 4.5 Opus",                    10.7, "critic"),
    ("GPT-5.2",                            10.5, "critic"),
    ("Gemini 2.5 Pro",                     10.4, "critic"),
    ("Gemini 2.5 Flash",                    8.2, "critic"),
    ("Gemini 2.5 Flash Lite",               4.5, "critic"),
    ("Cosmos Reason 1",                     2.8, "critic"),
    ("Cosmos Reason 2",                     1.2, "critic"),
]


def main():
    apply_style()
    # sort by flag rate, high to low, but keep the human reference pinned on top
    ref = [r for r in ROWS if r[2] == "ref"]
    rest = sorted([r for r in ROWS if r[2] != "ref"], key=lambda r: -r[1])
    rows = ref + rest

    n = len(rows)
    FIG_W, FIG_H = 6.5, 0.30 * n + 0.55
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([0.30, 0.16, 0.66, 0.80])

    ys = list(range(n))[::-1]           # first row at top
    for y, (label, val, kind) in zip(ys, rows):
        color = {"ref": LIGHT, "ours": ADOBE_RED, "critic": INK}[kind]
        ax.barh(y, val, height=0.62, color=color, edgecolor="none", zorder=3)
        # value label at the bar end
        ax.text(val + 1.0, y, f"{val:.1f}%",
                va="center", ha="left", fontsize=7.4,
                color=(ADOBE_RED if kind == "ours" else GRAY), zorder=4)

    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.6)
    # accent the two non-critic labels
    for tick, (_, _, kind) in zip(ax.get_yticklabels(), rows):
        if kind == "ours":
            tick.set_color(ADOBE_RED)
            tick.set_fontproperties(cm_bold(7.6))
        elif kind == "ref":
            tick.set_color(GRAY)

    ax.set_xlim(0, 60)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_xticklabels([f"{t}%" for t in [0, 10, 20, 30, 40, 50]], fontsize=7.0,
                       color=GRAY)
    ax.set_xlabel("physical-implausibility flag rate", fontsize=8.0, color=GRAY,
                  labelpad=3.0)

    ax.set_axisbelow(True)
    ax.grid(True, axis="x", color=GRID, lw=HAIR, zorder=0)
    ax.tick_params(length=0, colors=GRAY, labelcolor=INK)
    for name, sp in ax.spines.items():
        sp.set_visible(name == "bottom")
        sp.set_color(GRAY); sp.set_linewidth(HAIR)

    for ext in ("pdf", "png"):
        out = HERE / f"fig_physion_leaderboard.{ext}"
        fig.savefig(out, dpi=220 if ext == "png" else None,
                    metadata={"CreationDate": None} if ext == "pdf" else None)
        print(f"[write] {out.name}  {out.stat().st_size/1024:.0f} kB")
    plt.close(fig)


if __name__ == "__main__":
    main()
