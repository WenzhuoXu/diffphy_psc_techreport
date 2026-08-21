#!/usr/bin/env python3
"""Recall against cost for the published-evaluator comparison (was Table 6, tab:external).

One point per evaluator: horizontal axis is the mean number of model and tool calls per clip,
vertical axis is localized-flaw recall over the same 149 clips and the same 304 flaws. Up and to
the left is better --- more of the annotated flaws found for fewer calls. The point of the figure
is placement: PhyReAct sits above and to the left of the two structured baselines (it finds more
for less), while the single-pass VLM reaches nearly the same recall at one call but returns an
unstructured list with no per-verdict evidence.

Numbers are the frozen values of tab_external.tex, which analysis/emit_tex.py writes from the
run's own scored rows; they are transcribed here rather than recomputed, and the script prints
them so the page can be checked against the table. The full table, with the output each evaluator
produces and the not-evaluated row, is kept in the appendix.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

OUT = Path(__file__).resolve().parent

# muted journal palette (as in the reference figure)
INK, GRAY = "#2A2A2A", "#6E6E6E"
GRID = "#E4E4E4"
BLUE = "#5B7FA6"      # structured baselines
ROSE = "#A65461"      # free-text baseline
GREEN = "#3E7D5A"     # PhyReAct -- the accent / "final" role
TITLE = "#4A5A78"     # in-plot title, muted slate
HAIRLINE = 0.7

# (label, found, recall %, calls/clip, kind) -- kind picks the marker.
#   "ours"  : PhyReAct, the localized evidence-carrying critic (filled star)
#   "struct": per-question structured baselines (open circle)
#   "free"  : free-text flaw list (open square)
# Values are the frozen entries of tab_external.tex.
POINTS = [
    ("PhyReAct",                    228, 75.0, 14.1, "ours"),
    ("Question decomposition",      164, 53.9, 21.9, "struct"),
    ("Davidsonian scene graph",     161, 53.0, 52.1, "struct"),
    ("Single-pass VLM, same model", 222, 73.0,  1.0, "free"),
]
# Modular video QA (proviq/morevqa) was not evaluated under this protocol -> off-plot note.
NOT_EVALUATED = "Modular video QA: not evaluated under this protocol"

# short label beside each point (name + recall), and its offset placement.
SHORT = {
    "PhyReAct":                    "PhyReAct  75.0%",
    "Single-pass VLM, same model": "Single-pass VLM  73.0%",
    "Question decomposition":      "Question decomp.  53.9%",
    "Davidsonian scene graph":     "Scene graph  53.0%",
}
PLACEMENT = {
    "PhyReAct":                    ((10,   4), "left",   "bottom"),
    "Single-pass VLM, same model": ((10,   0), "left",   "center"),
    "Question decomposition":      ((0,  -12), "center", "top"),
    "Davidsonian scene graph":     ((-8,  10), "right",  "bottom"),
}


def use_source_serif() -> None:
    """Match the report body face: register Source Serif from wherever it lives and
    set it as the figure's default family (serif), like the other data figures."""
    import glob
    roots = [
        "/usr/local/texlive/2024/texmf-dist/fonts/opentype/adobe/sourceserifpro",
        str(Path.home() / "Library/Caches/Tectonic/bundles/data/*"),
        str(Path.home() / "Library/Fonts"), "/Library/Fonts",
        "/System/Library/Fonts/Supplemental",
    ]
    for root in roots:
        for base in glob.glob(root):
            for f in glob.glob(str(Path(base) / "SourceSerif*.otf")) + \
                     glob.glob(str(Path(base) / "SourceSerif*.ttf")):
                try:
                    font_manager.fontManager.addfont(f)
                except Exception:
                    pass
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Source Serif 4", "Source Serif Pro", "DejaVu Serif"],
    })


def marker(kind: str) -> dict:
    """Solid dots with a white edge (reference style); PhyReAct is the larger accent."""
    if kind == "ours":
        return dict(marker="o", ms=13, mfc=GREEN, mec="white", mew=1.3, zorder=6)
    if kind == "free":
        return dict(marker="o", ms=10, mfc=ROSE, mec="white", mew=1.1, zorder=5)
    return dict(marker="o", ms=10, mfc=BLUE, mec="white", mew=1.1, zorder=5)


def main() -> int:
    use_source_serif()
    fig, ax = plt.subplots(figsize=(5.6, 3.7))   # printed at ~0.72 linewidth

    ax.set_axisbelow(True)
    ax.grid(True, which="major", color=GRID, linewidth=0.8, zorder=0)

    for name, found, recall, calls, kind in POINTS:
        ax.plot([calls], [recall], **marker(kind))
        (dx, dy), ha, va = PLACEMENT[name]
        col = GREEN if kind == "ours" else INK
        wt = "bold" if kind == "ours" else "normal"
        ax.annotate(SHORT[name], (calls, recall), textcoords="offset points",
                    xytext=(dx, dy), fontsize=8.4, color=col, ha=ha, va=va, weight=wt)

    ax.set_xlim(-3.5, 58)
    ax.set_ylim(47, 82)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_yticks([50, 55, 60, 65, 70, 75, 80])
    ax.set_xlabel("mean model / tool calls per clip", fontsize=9.0, color=INK)
    ax.set_ylabel("localized-flaw recall (%)  of 304", fontsize=9.0, color=INK)

    # in-plot title, muted slate, upper-left (reference style)
    ax.text(0.02, 0.965, "Recall against cost per clip", transform=ax.transAxes,
            fontsize=10.5, color=TITLE, weight="bold", ha="left", va="top")
    ax.text(0.02, 0.895, "up and to the left is better", transform=ax.transAxes,
            fontsize=8.0, color=GRAY, style="italic", ha="left", va="top")

    ax.annotate(NOT_EVALUATED, (0.0, -0.155), xycoords="axes fraction", fontsize=7.8,
                color=GRAY, ha="left", va="top")

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(HAIRLINE)
        ax.spines[s].set_color(GRAY)
    ax.tick_params(labelsize=8.3, colors=GRAY, width=HAIRLINE, length=2.5)

    # bottom horizontal legend (reference style): one row, marker + meaning
    handles = [
        Line2D([], [], linestyle="none", marker="o", ms=9, mfc=GREEN, mec="white", mew=1.2,
               label="PhyReAct (localized, evidence-carrying)"),
        Line2D([], [], linestyle="none", marker="o", ms=8, mfc=BLUE, mec="white", mew=1.0,
               label="per-question (structured)"),
        Line2D([], [], linestyle="none", marker="o", ms=8, mfc=ROSE, mec="white", mew=1.0,
               label="free-text flaw list"),
    ]
    leg = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.20),
                    ncol=3, frameon=False, fontsize=8.0, handletextpad=0.4,
                    columnspacing=1.4, borderaxespad=0.0)
    for t in leg.get_texts():
        t.set_color(INK)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_external.{ext}", dpi=220, bbox_inches="tight")
        print(f"[write] {OUT / f'fig_external.{ext}'}")
    for name, found, recall, calls, kind in POINTS:
        print(f"  {name:<30} found {found:>3}  recall {recall:5.1f}%  calls {calls:>5.1f}  [{kind}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
