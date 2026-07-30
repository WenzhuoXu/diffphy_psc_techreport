#!/usr/bin/env python3
"""fig_plan_chain -- how a claim about time is chained to measurements on the video.

Every number and every frame here is real: the plan line is the one the planner
emitted for this clip, the two windows are the ones the event-timing measurements
returned, the arithmetic is the compiler's own, and the frames are decoded from the
clip at the times the windows name.

    python3 make_fig_plan_chain.py
"""
import glob
import os
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.image as mpimg

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets" / "plan_graph"
CLIP = Path("/Users/wenzhuox/diffphy_exp013/scripts/exp013/exp029/design/"
            "temporal_page/clips/b26071d7-b0fa-533e-90da-2fd3ee1924a9.mp4")

RED, INK, GRAY = "#FA0F00", "#1A1A1A", "#6E6E6E"
L1, L2 = "#E5E5E5", "#F5F5F5"
HAIR = 0.6

# ---- the measured trace (verbatim from the recorded run) --------------------
CLIP_DUR = 8.0
EV_A = ("the first domino tips forward", 1.62, 1.67)
EV_B = ("the first domino triggers a chain reaction", 2.33, 7.96)
TOL = 0.01                      # 25% x min(window duration)
PLAN_LINE = ('check(c1, before(window("the first domino tips forward"),\n'
             '                 window("the first domino triggers a chain reaction")))')
FRAME_TIMES = [0.60, 1.65, 2.33, 5.00, 7.90]
FRAME_NOTE = {1.65: "A", 2.33: "B"}


def fonts():
    roots = ["/Users/wenzhuox/Library/Caches/Tectonic/bundles/data/*",
             str(Path.home() / "Library/Fonts"), "/Library/Fonts",
             "/Library/Application Support/Adobe/*/*/Fonts"]
    n = 0
    for r in roots:
        for base in glob.glob(r):
            for pat in ("SourceSansPro-*.otf", "SourceSerifPro-*.otf", "SourceCodePro-*.otf"):
                for f in glob.glob(os.path.join(base, pat)):
                    try:
                        font_manager.fontManager.addfont(f); n += 1
                    except Exception:
                        pass
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Source Sans Pro", "Source Sans 3", "DejaVu Sans"],
        "font.serif": ["Source Serif Pro", "Source Serif 4", "DejaVu Serif"],
        "font.monospace": ["Source Code Pro", "DejaVu Sans Mono"],
        "pdf.fonttype": 42, "text.color": INK, "axes.edgecolor": GRAY,
    })
    have = {f.name for f in font_manager.fontManager.ttflist}
    print(f"[fonts] registered {n}; Source Sans Pro present: {'Source Sans Pro' in have}")


def grab_frames():
    ASSETS.mkdir(parents=True, exist_ok=True)
    out = {}
    for t in FRAME_TIMES:
        p = ASSETS / f"f_{t:.2f}.png"
        if not p.exists():
            subprocess.run(["/opt/homebrew/bin/ffmpeg", "-v", "error", "-ss", str(t),
                            "-i", str(CLIP), "-frames:v", "1", "-vf", "scale=320:-1",
                            "-y", str(p)], check=True)
        out[t] = mpimg.imread(p)
        print(f"[frame] t={t:.2f}s -> {p.name} {out[t].shape[1]}x{out[t].shape[0]}")
    return out


def box(ax, x, y, w, h, fc="white", ec=INK, lw=HAIR, r=0.012):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=fc, edgecolor=ec, linewidth=lw,
                                transform=ax.transAxes, clip_on=False, zorder=2))


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), xycoords=ax.transAxes,
                textcoords=ax.transAxes, zorder=3,
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=0.55,
                                shrinkA=0, shrinkB=0, mutation_scale=8))


def main():
    fonts()
    frames = grab_frames()

    FIG_W, FIG_H = 6.5, 4.75
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # ---------------- 1. the plan line the planner wrote --------------------
    ax.text(0.012, 0.975, "the plan line the planner wrote, before any frame is read",
            fontsize=7.4, color=GRAY, va="top", ha="left")
    box(ax, 0.012, 0.878, 0.976, 0.082, fc=L2, ec=GRAY, lw=0.5)
    ax.text(0.026, 0.938, PLAN_LINE, fontsize=6.9, family="monospace",
            color=INK, va="top", ha="left", linespacing=1.5)

    # ---------------- 2. the chain -----------------------------------------
    ytop, hbox = 0.760, 0.088
    box(ax, 0.012, ytop, 0.238, hbox, fc="#EDEDED", ec=INK)
    ax.text(0.131, ytop + hbox / 2, "a claim about\nwhen things happen",
            fontsize=7.6, ha="center", va="center", color=INK, linespacing=1.35)

    box(ax, 0.300, ytop, 0.210, hbox, fc="white", ec=RED)
    ax.text(0.405, ytop + hbox / 2, "one ordering check\n(before, with slack)",
            fontsize=7.6, ha="center", va="center", color=INK, linespacing=1.35)

    box(ax, 0.560, ytop + 0.050, 0.428, 0.052, fc="white", ec=INK)
    ax.text(0.574, ytop + 0.076, "when does A happen?", fontsize=7.4, ha="left",
            va="center", color=INK)
    box(ax, 0.560, ytop - 0.010, 0.428, 0.052, fc="white", ec=INK)
    ax.text(0.574, ytop + 0.016, "when does B happen?", fontsize=7.4, ha="left",
            va="center", color=INK)

    arrow(ax, 0.250, ytop + hbox / 2, 0.300, ytop + hbox / 2)
    arrow(ax, 0.510, ytop + hbox / 2, 0.560, ytop + 0.076)
    arrow(ax, 0.510, ytop + hbox / 2, 0.560, ytop + 0.016)

    # the gate that has to pass first
    ax.text(0.300, ytop - 0.036,
            "first, both events must be confirmed to occur by the verifier that reads "
            "the frames;\nonly then do the two windows decide the order",
            fontsize=7.0, color=GRAY, ha="left", va="top", linespacing=1.4)

    # ---------------- 3. the timeline with the measured windows -------------
    tl_l, tl_r = 0.075, 0.975
    tl_y = 0.560
    def tx(t): return tl_l + (t / CLIP_DUR) * (tl_r - tl_l)

    ax.text(0.012, tl_y + 0.088, "what the two measurements returned, on the clip's own clock",
            fontsize=7.4, color=GRAY, va="bottom", ha="left")

    ax.add_patch(Rectangle((tl_l, tl_y), tl_r - tl_l, 0.030, facecolor=L2,
                           edgecolor=GRAY, lw=0.5, transform=ax.transAxes,
                           clip_on=False, zorder=1))
    for t in range(0, 9):
        ax.plot([tx(t), tx(t)], [tl_y - 0.008, tl_y], color=GRAY, lw=0.5,
                transform=ax.transAxes, clip_on=False)
        ax.text(tx(t), tl_y - 0.014, f"{t}", fontsize=6.6, color=GRAY,
                ha="center", va="top")
    ax.text(tl_r, tl_y - 0.036, "seconds", fontsize=6.6, color=GRAY, ha="right", va="top")

    # window A (narrow) and window B (wide)
    for (name, a, b), col, lab, yoff in ((EV_A, RED, "A", 0.040), (EV_B, INK, "B", 0.040)):
        w = max(tx(b) - tx(a), 0.004)
        ax.add_patch(Rectangle((tx(a), tl_y), w, 0.030, facecolor=col, alpha=0.30,
                               edgecolor=col, lw=0.7, transform=ax.transAxes,
                               clip_on=False, zorder=3))
        ax.text(tx(a) + w / 2, tl_y + yoff, f"{lab}  [{a:.2f}, {b:.2f}] s",
                fontsize=6.9, color=col, ha="center", va="bottom")
    ax.text(tl_l, tl_y + 0.062, f"A  {EV_A[0]}", fontsize=6.9, color=RED, ha="left", va="bottom")
    ax.text(tl_r, tl_y + 0.062, f"B  {EV_B[0]}", fontsize=6.9, color=INK, ha="right", va="bottom")

    # ---------------- 4. the frames, under the times they name --------------
    fw = 0.148
    fy = 0.235
    placed = []
    for t in FRAME_TIMES:
        img = frames[t]
        cx = tx(t)
        x0 = min(max(cx - fw / 2, 0.006), 0.994 - fw)
        for prev in placed:                       # never overlap a neighbour
            if x0 < prev + fw + 0.010:
                x0 = prev + fw + 0.010
        x0 = min(x0, 0.994 - fw)
        placed.append(x0)
        fh = fw * (img.shape[0] / img.shape[1]) * (FIG_W / FIG_H)
        axi = fig.add_axes([x0, fy, fw, fh])
        axi.imshow(img); axi.set_xticks([]); axi.set_yticks([])
        for sp in axi.spines.values():
            sp.set_color(RED if t in FRAME_NOTE else GRAY)
            sp.set_linewidth(0.8 if t in FRAME_NOTE else HAIR)
        axi.set_title(f"{t:.2f} s" + (f"  ({FRAME_NOTE[t]} starts)" if t in FRAME_NOTE else ""),
                      fontsize=6.4, color=RED if t in FRAME_NOTE else GRAY, pad=2.0)
        # leader from the frame up to its instant on the timeline
        fig.add_artist(plt.Line2D([x0 + fw / 2, cx], [fy + fh + 0.030, tl_y],
                                  transform=fig.transFigure, color=L1, lw=0.5, zorder=0))

    # ---------------- 5. the arithmetic the compiler applied ----------------
    box(ax, 0.012, 0.020, 0.976, 0.150, fc=L2, ec=GRAY, lw=0.5)
    ax.text(0.026, 0.150,
            "how the order was decided", fontsize=7.4, color=GRAY, va="top", ha="left")
    ax.text(0.026, 0.118,
            f"tolerance = 25% of the shorter window = {TOL:.2f} s.   "
            f"A before B iff A_end ≤ B_start + tolerance.\n"
            f"{EV_A[2]:.2f} ≤ {EV_B[1]:.2f} + {TOL:.2f}   →   yes, so the claim is supported.",
            fontsize=7.0, color=INK, va="top", ha="left", linespacing=1.25)
    ax.text(0.026, 0.052,
            "if neither direction passes, the windows carry no order and the claim "
            "remains unresolved.",
            fontsize=6.9, color=GRAY, va="top", ha="left")

    for ext in ("pdf", "png"):
        out = HERE / f"fig_plan_chain.{ext}"
        fig.savefig(out, dpi=200 if ext == "png" else None,
                    metadata={"CreationDate": None} if ext == "pdf" else None)
        print(f"[write] {out.name}  {out.stat().st_size/1024:.0f} kB")
    plt.close(fig)


if __name__ == "__main__":
    main()
