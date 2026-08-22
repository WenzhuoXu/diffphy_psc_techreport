#!/usr/bin/env python3
"""fig_refine_data -- the refinement loop read two ways, on the same 589 clips.

Two panels making one honest point by contrast:

  Left  -- PhyReAct's OWN physical metric (the quantity the rewrite optimizes):
           horizontal bars for the share of flagged clips that improve, the share
           whose verdict flips implausible->plausible, and the collateral rate on
           unflagged clips. On its own metric the loop moves a lot.

  Right -- an INDEPENDENT rater (VideoPhy-2's AutoRater): a dumbbell per threshold,
           baseline (navy) to rewrite (red). The dumbbells collapse to points --- no
           movement --- on both the full set and the hard subset, in the semantic and
           the physical dimension.

Every number is transcribed verbatim from tab_refine_selfmetric.tex and
tab_refine_autorater.tex; nothing is recomputed here.

    python3 make_fig_refine_data.py
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent

# ---- report visual system: plot-appropriate navy + brick red ---------------- #
NAVY = "#20456E"   # baseline / structure
RED = "#C42E2E"    # the rewrite: a legible brick red, not a saturated screen red
INK = "#1A1A1A"
GRAY = "#6E6E6E"
MUTE = "#B7BCC2"   # collateral: a recessive gray (an unintended, small effect)
GRID = "#E9E9E9"
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
        "axes.formatter.use_mathtext": True,
    })


def cm_bold(size):
    return fm.FontProperties(fname=CM_BOLD, size=size)


# ---- data (verbatim from the two refine tables) ---------------------------- #
# Left: (label, fraction_text, bar_percent, label_percent_text, colour)
SELF = [
    ("improves on flagged",              "63/87",  72.4, "72.4", RED),
    ("flips impl.$\\to$plaus.",          "49/65",  75.4, "75",   RED),
    ("collateral (unflagged worse)",     "8/502",  1.6,  "1.6",  MUTE),
]
P_BASE, P_REWRITE = 0.797, 0.862   # overall mean p_plausible

# Right: threshold -> {All:(base,rewrite), Hard:(base,rewrite)}, in percent
AUTORATER = [
    ("SA $\\geq 4$", (27.8, 28.0), (8.9, 7.9)),
    ("PC $\\geq 4$", (55.0, 54.5), (37.2, 36.2)),
    ("Joint",        (22.8, 23.1), (3.9, 3.4)),
]


def panel_self(ax):
    ys = list(range(len(SELF)))[::-1]
    for y, (label, frac, pct, pct_txt, col) in zip(ys, SELF):
        ax.barh(y, pct, height=0.56, color=col, edgecolor="none", zorder=3)
        ax.text(pct + 2.5, y, f"{frac}  ({pct_txt}%)",
                va="center", ha="left", fontsize=7.6,
                color=(INK if col is RED else GRAY), zorder=4)
    ax.set_yticks(ys)
    ax.set_yticklabels([s[0] for s in SELF], fontsize=8.0)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels([f"{t}%" for t in (0, 25, 50, 75, 100)], fontsize=7.2, color=GRAY)
    ax.set_ylim(-0.6, len(SELF) - 0.35)
    ax.set_title("PhyReAct's own physical metric", fontproperties=cm_bold(7.8),
                 color=INK, pad=5, loc="left")
    ax.set_axisbelow(True)
    ax.grid(True, axis="x", color=GRID, lw=HAIR, zorder=0)
    ax.tick_params(length=0, colors=GRAY, labelcolor=INK)
    for name, sp in ax.spines.items():
        sp.set_visible(name == "bottom")
        sp.set_color(GRAY); sp.set_linewidth(HAIR)


def panel_autorater(ax):
    # rows top->bottom: All (3 thresholds) then Hard (3), each a baseline->rewrite dumbbell
    rows = []                       # (y, base, rewrite, thr_label)
    y = 0
    groups = [("All ($n=589$)", 1), ("Hard subset", 2)]
    ytop = 2 * len(AUTORATER) + 1   # slot count with a gap between groups
    slot = ytop
    group_spans = []
    for gname, gi in groups:
        start = slot
        for thr, allv, hardv in AUTORATER:
            slot -= 1
            b, r = (allv if gi == 1 else hardv)
            rows.append((slot, b, r, thr))
        group_spans.append((gname, start - 1, slot))
        slot -= 1                   # blank gap between groups

    for yy, b, r, thr in rows:
        ax.plot([b, r], [yy, yy], color=GRAY, lw=1.0, zorder=2, solid_capstyle="round")
        ax.plot(b, yy, "o", ms=5.2, color=NAVY, zorder=3)
        ax.plot(r, yy, "o", ms=5.2, color=RED, zorder=4)

    ax.set_yticks([r[0] for r in rows])
    ax.set_yticklabels([r[3] for r in rows], fontsize=8.0)
    # group labels just left of the threshold tick labels
    for gname, ghi, glo in group_spans:
        ax.text(-0.235, (ghi + glo) / 2.0, gname, transform=ax.get_yaxis_transform(),
                fontsize=7.4, color=GRAY, ha="center", va="center", rotation=90,
                fontproperties=cm_bold(7.4))

    ax.set_xlim(0, 62)
    ax.set_xticks([0, 20, 40, 60])
    ax.set_xticklabels([f"{t}%" for t in (0, 20, 40, 60)], fontsize=7.2, color=GRAY)
    ax.set_ylim(min(r[0] for r in rows) - 0.7, max(r[0] for r in rows) + 0.7)
    ax.set_xlabel("clips passing the threshold", fontsize=8.0, color=GRAY, labelpad=2.5)
    ax.set_title("independent rater (VideoPhy-2 AutoRater)", fontproperties=cm_bold(7.8),
                 color=INK, pad=5, loc="left")
    ax.set_axisbelow(True)
    ax.grid(True, axis="x", color=GRID, lw=HAIR, zorder=0)
    ax.tick_params(length=0, colors=GRAY, labelcolor=INK)
    for name, sp in ax.spines.items():
        sp.set_visible(name == "bottom")
        sp.set_color(GRAY); sp.set_linewidth(HAIR)

    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=NAVY,
                      markersize=5.2, label="baseline"),
               Line2D([0], [0], marker="o", color="none", markerfacecolor=RED,
                      markersize=5.2, label="rewrite")]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=7.4,
              handletextpad=0.3, borderaxespad=0.3, labelspacing=0.25)


def main():
    apply_style()
    fig = plt.figure(figsize=(6.6, 2.35))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.46,
                          left=0.225, right=0.985, top=0.86, bottom=0.17)
    panel_self(fig.add_subplot(gs[0, 0]))
    panel_autorater(fig.add_subplot(gs[0, 1]))
    for ext in ("pdf", "png"):
        out = HERE / f"fig_refine_data.{ext}"
        fig.savefig(out, dpi=220 if ext == "png" else None,
                    metadata={"CreationDate": None} if ext == "pdf" else None)
        print(f"[write] {out.name}  {out.stat().st_size/1024:.0f} kB")
    plt.close(fig)


if __name__ == "__main__":
    main()
