#!/usr/bin/env python3
"""fig_plan_chain (Figure 3) -- how a claim about time is chained to measurements.

A redesigned, fully reproducible temporal trace. The top row is the verification
pipeline read left to right --- PLAN, GATE, MEASURE, DECIDE --- and the bottom row
is the evidence it stands on: a clip timeline carrying the two measured windows and
a strip of real frames decoded at the instants those windows name.

Every number and every frame is real. The plan line is the one the planner emitted
for this clip, the two windows are the ones the event-timing measurements returned,
the tolerance is DERIVED from the paper's own quarter-window rule (eq:temporal-before),
and the frames come from the clip at the times the windows name.

Text matches the report body face (Source Serif); only the plan line stays
monospace, exactly as code is set in the body.

Reproducibility: reads the clip from the in-repo asset
(assets/n-G4vR5pcmA_ljrqp3R52g.mp4) and discovers ffmpeg and the report's Source
fonts on whatever machine it runs on, so the figure can be re-rendered and tracked
without any per-machine paths. Override the clip with PLAN_CHAIN_CLIP if it moves.

    python3 make_fig_plan_chain.py

Idempotent: re-running overwrites fig_plan_chain.{pdf,png} and reuses cached frames.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.image as mpimg

# --------------------------------------------------------------------------- #
# paths -- everything resolves relative to the report root, no per-machine dirs
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent          # .../figs
REPORT = HERE.parent                            # report root
CLIP = Path(os.environ.get(
    "PLAN_CHAIN_CLIP",
    REPORT / "assets" / "n-G4vR5pcmA_ljrqp3R52g.mp4",
)).expanduser()
FRAME_CACHE = HERE / "assets" / "plan_chain"    # decoded frames, regenerable

# --------------------------------------------------------------------------- #
# report visual system -- single accent, ink, one keyline grey, two fill greys
# --------------------------------------------------------------------------- #
RED, INK, GRAY = "#FA0F00", "#1A1A1A", "#6E6E6E"
L1, L2 = "#E5E5E5", "#F5F5F5"                    # keyline fill, panel fill
HAIR = 0.6

# --------------------------------------------------------------------------- #
# the measured trace (verbatim from the recorded run of this clip)
# --------------------------------------------------------------------------- #
CLIP_DUR = 8.0                                   # timeline max; asserted vs clip
EV_A = ("the first domino tips forward", 1.62, 1.67)
EV_B = ("the first domino triggers a chain reaction", 2.33, 7.96)
PLAN_LINE = ('check(c1, before(window("the first domino tips forward"),\n'
             '                 window("the first domino triggers a chain reaction")))')
FRAME_TIMES = [0.60, 1.65, 2.33, 5.00, 7.90]
FRAME_NOTE = {1.65: "A", 2.33: "B"}              # which frames anchor a window start


def temporal_slack(a, b):
    """delta(A,B) = 1/4 * min(|W_A|, |W_B|), the paper's quarter-window rule
    (eq:temporal-before). Derived, never hand-entered."""
    return 0.25 * min(a[2] - a[1], b[2] - b[1])


TOL = temporal_slack(EV_A, EV_B)


# --------------------------------------------------------------------------- #
# fonts -- register the report's Source faces from wherever they live; the body
# face is Source Serif, so that is what the figure's prose uses. The plan line
# stays monospace (Source Code Pro), matching how code is set in the body.
# --------------------------------------------------------------------------- #
FONT_DIRS = [
    str(Path.home() / "Library/Caches/Tectonic/bundles/data/*"),  # the report's own OTFs
    str(Path.home() / "Library/Fonts"),
    "/Library/Fonts",
    "/Library/Application Support/Adobe/*/*/*/*/*/fonts",          # Acrobat bundle
    "/Library/Application Support/Adobe/*/*/Fonts",
    "/System/Library/Frameworks/Ruby.framework/Versions/*/usr/lib/ruby/*/rdoc/generator/template/darkfish/fonts",
]
FONT_PATTERNS = ("SourceSans*.otf", "SourceSans*.ttf",
                 "SourceSerif*.otf", "SourceSerif*.ttf",
                 "SourceCode*.otf", "SourceCode*.ttf")

# The report body is Computer Modern: prose in CMR10, code (\texttt) in CM
# typewriter. matplotlib ships the real cmr10/cmtt10 TTFs, so this figure's text
# is glyph-for-glyph the body font rather than a look-alike.
SERIF = ["cmr10", "CMU Serif", "DejaVu Serif"]
SANS = ["cmss10", "DejaVu Sans"]
MONO = ["cmtt10", "DejaVu Sans Mono"]


def fonts():
    # cmr10/cmtt10/cmss10 ship with matplotlib and are already registered by name.
    plt.rcParams.update({
        "font.family": "serif",           # body face: prose is Computer Modern serif
        "font.serif": SERIF,
        "font.sans-serif": SANS,
        "font.monospace": MONO,
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,      # cmr10 has no U+2212; mathtext handles minus
        "axes.formatter.use_mathtext": True,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "text.color": INK, "axes.edgecolor": GRAY,
    })
    for label, fams in (("serif", SERIF), ("mono", MONO)):
        resolved = font_manager.findfont(
            font_manager.FontProperties(family=fams), fallback_to_default=True)
        print(f"[fonts] {label:<5s} -> {Path(resolved).name} "
              f"({font_manager.FontProperties(fname=resolved).get_name()})")


# --------------------------------------------------------------------------- #
# clip -- probe it, assert the recorded trace fits, then decode the frames
# --------------------------------------------------------------------------- #
def _ffbin(name):
    p = shutil.which(name) or f"/opt/homebrew/bin/{name}"
    if not Path(p).exists():
        raise SystemExit(f"{name} not found on PATH or /opt/homebrew/bin; "
                         f"install ffmpeg to decode {CLIP.name}")
    return p


def probe_clip():
    if not CLIP.exists():
        raise SystemExit(f"clip not found: {CLIP}\nset PLAN_CHAIN_CLIP to its location.")
    out = subprocess.run(
        [_ffbin("ffprobe"), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate,nb_frames:format=duration",
         "-of", "json", str(CLIP)],
        check=True, capture_output=True, text=True).stdout
    info = json.loads(out)
    st = info["streams"][0]
    num, den = (float(x) for x in st["r_frame_rate"].split("/"))
    dur = float(info["format"]["duration"])
    fps = num / den if den else float("nan")
    print(f"[clip] {CLIP.name}  {st['width']}x{st['height']}  "
          f"{dur:.3f}s  {fps:.3f} fps  {st.get('nb_frames','?')} frames")
    assert abs(dur - CLIP_DUR) < 0.25, f"clip duration {dur:.3f}s != {CLIP_DUR}s"
    for name, a, b in (EV_A, EV_B):
        assert 0 <= a <= b <= dur + 1e-6, f"window [{a},{b}] outside clip ({name})"
    for t in FRAME_TIMES:
        assert 0 <= t <= dur + 1e-6, f"frame time {t}s outside clip"
    return dur


def grab_frames():
    FRAME_CACHE.mkdir(parents=True, exist_ok=True)
    ffmpeg = _ffbin("ffmpeg")
    out = {}
    for t in FRAME_TIMES:
        p = FRAME_CACHE / f"f_{t:.2f}.png"
        if not p.exists():
            subprocess.run([ffmpeg, "-v", "error", "-ss", str(t),
                            "-i", str(CLIP), "-frames:v", "1", "-vf", "scale=360:-1",
                            "-y", str(p)], check=True)
        out[t] = mpimg.imread(p)
        print(f"[frame] t={t:.2f}s -> {p.name} {out[t].shape[1]}x{out[t].shape[0]}")
    return out


# --------------------------------------------------------------------------- #
# drawing helpers (all in axis fraction coordinates on a single overlay axes)
# --------------------------------------------------------------------------- #
def panel(ax, x, y, w, h, fc="white", ec=INK, lw=HAIR, r=0.014, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=fc, edgecolor=ec, linewidth=lw,
                                transform=ax.transAxes, clip_on=False, zorder=z))


def flow_arrow(ax, x0, x1, y, color=GRAY, lw=1.1):
    """A short horizontal connector between two pipeline cards."""
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y),
                 transform=ax.transAxes, clip_on=False, zorder=4,
                 arrowstyle="-|>", mutation_scale=11,
                 shrinkA=0, shrinkB=0, color=color, lw=lw))


def leader(fig, x0, y0, x1, y1, color=L1, lw=0.55):
    fig.add_artist(plt.Line2D([x0, x1], [y0, y1], transform=fig.transFigure,
                              color=color, lw=lw, zorder=0))


def elbow_leader(ax, x_frame, y_frame, x_time, y_rail, y_time, color=L1, lw=0.5):
    """A tidy connector: straight up from the frame to a shared rail just below
    the timeline, then a short diagonal to the true instant. Keeps the long run
    vertical so the leaders do not slant across the windows."""
    ax.plot([x_frame, x_frame], [y_frame, y_rail], color=color, lw=lw,
            transform=ax.transAxes, clip_on=False, zorder=0)
    ax.plot([x_frame, x_time], [y_rail, y_time], color=color, lw=lw,
            transform=ax.transAxes, clip_on=False, zorder=0)


def main():
    fonts()
    probe_clip()
    frames = grab_frames()
    verdict = "supported" if EV_A[2] <= EV_B[1] + TOL else "unresolved"
    print(f"[trace] delta = 1/4 * min({EV_A[2]-EV_A[1]:.2f}, {EV_B[2]-EV_B[1]:.2f}) "
          f"= {TOL:.4f} s")
    print(f"[trace] A_end {EV_A[2]:.2f} <= B_start {EV_B[1]:.2f} + {TOL:.4f} -> {verdict}")

    # Compact canvas: shorter than before so the timeline and the filmstrip sit
    # close together, with no tall empty band of leaders between them. Because all
    # y-coordinates are fractions of this height, the bands are packed tightly
    # below (see the y-values) rather than relying on the canvas height alone.
    FIG_W, FIG_H = 6.5, 3.35
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    AR = FIG_W / FIG_H                                  # for square-ish frames

    # A quiet caption above each band: sentence case, regular weight, body serif.
    def kicker(x, y, text):
        ax.text(x, y, text, fontsize=7.0, color=GRAY, va="top", ha="left")

    # ===================================================================== #
    # 0. the plan line, set as a quoted listing: no fill box, just a thin accent
    #    rule down the left margin.
    # ===================================================================== #
    kicker(0.014, 0.986, "The plan line the planner wrote, before any frame is read")
    ax.plot([0.017, 0.017], [0.905, 0.958], color=RED, lw=1.6,
            transform=ax.transAxes, clip_on=False, zorder=3)
    ax.text(0.036, 0.953, PLAN_LINE, fontsize=6.7, family="monospace",
            color=INK, va="top", ha="left", linespacing=1.5)

    # ===================================================================== #
    # 1. the pipeline: four stages read left to right, each a clean card with a
    #    numbered title, a hairline under it, and short serif body lines.
    # ===================================================================== #
    kicker(0.014, 0.884, "How one ordering check is carried out, left to right")

    # Flat cards: every line (title, body, data, outcome) is packed from the top
    # on one pitch, so there is no empty band in the middle and the card height is
    # only as tall as its content. All four share one height for a clean row.
    xs = [0.014, 0.266, 0.514, 0.798]                  # card left edges
    ws = [0.212, 0.210, 0.242, 0.188]                  # card widths
    names = ["Plan", "Gate", "Measure", "Decide"]
    edges = [INK, INK, INK, RED]                       # Decide is the accented outcome

    tx0 = 0.014                                        # text inset inside a card
    ytop = 0.812                                       # card top (lowered to clear the kicker above)
    PITCH = 0.0300                                     # baseline-to-baseline
    y_rule = ytop - 0.052                              # hairline BELOW the title (clears the glyphs)
    y_body0 = ytop - 0.074                             # first (top) body baseline
    NBODY = 4                                          # max text lines below the rule
    # foot sits below the LAST baseline by a full line's descent + margin, so even
    # a 4th line clears the bottom edge; the card is still flat (no mid gap).
    y_last = y_body0 - (NBODY - 1) * PITCH             # 4th (lowest) baseline
    ybot = y_last - 0.034
    row_h = ytop - ybot

    body = dict(fontsize=7.0, color=INK)
    small = dict(fontsize=6.3, color=GRAY, style="italic")
    mono = dict(fontsize=6.7, family="monospace")

    def card_lines(x, rows):
        """Lay rows top-down from y_body0 on the shared pitch."""
        for i, (text, kw) in enumerate(rows):
            ax.text(x, y_body0 - i * PITCH, text, va="top", ha="left", **kw)

    for x, w, nm, ec in zip(xs, ws, names, edges):
        acc = RED if ec == RED else GRAY
        panel(ax, x, ybot, w, row_h, fc="white", ec=ec, lw=(1.0 if ec == RED else HAIR))
        num = names.index(nm) + 1
        ax.text(x + tx0, ytop - 0.016, f"{num}", fontsize=8.4, color=acc,
                va="top", ha="left")
        ax.text(x + tx0 + 0.025, ytop - 0.018, nm, fontsize=7.8, color=INK,
                va="top", ha="left")
        ax.plot([x + tx0, x + w - tx0], [y_rule, y_rule],
                color=L1, lw=0.6, transform=ax.transAxes, clip_on=False, zorder=3)

    # 1 Plan
    card_lines(xs[0] + tx0, [
        ("reads the claim as an", body),
        ("ordering A $\\rightarrow$ B,", body),
        ("carried with slack", body),
    ])

    # 2 Gate
    card_lines(xs[1] + tx0, [
        ("the verifier confirms", body),
        ("both A and B occur", body),
        ("occurrence gates order", small),
    ])

    # 3 Measure -- prose, then the two returned windows in machine type
    card_lines(xs[2] + tx0, [
        ("event timing returns", body),
        ("one window per event:", body),
        ("A = [1.62, 1.67] s", dict(color=RED, **mono)),
        ("B = [2.33, 7.96] s", dict(color=INK, **mono)),
    ])

    # 4 Decide -- prose, then the accented outcome (on the 3rd line, not the 4th,
    # so the larger red type never rides the bottom edge)
    card_lines(xs[3] + tx0, [
        ("the predicate resolves", body),
        ("to a state, with", body),
    ])
    ax.text(xs[3] + tx0, y_body0 - 2 * PITCH, "$\\rightarrow$ supported", fontsize=8.4,
            color=RED, va="top", ha="left")

    # connectors between the cards
    for i in range(3):
        flow_arrow(ax, xs[i] + ws[i] + 0.003, xs[i + 1] - 0.003, ybot + row_h / 2)

    # ===================================================================== #
    # 2. the evidence base: timeline + the two measured windows
    # ===================================================================== #
    tl_l, tl_r = 0.075, 0.965
    tl_y = 0.478
    tl_h = 0.032
    def tx(t): return tl_l + (t / CLIP_DUR) * (tl_r - tl_l)

    # The two window names sit on their own line above the bar (left and right);
    # no separate caption is needed -- the bar and frames are self-explanatory.
    ax.text(tl_l, tl_y + tl_h + 0.034, f"A: {EV_A[0]}",
            fontsize=7.0, color=RED, ha="left", va="bottom")
    ax.text(tl_r, tl_y + tl_h + 0.034, f"B: {EV_B[0]}",
            fontsize=7.0, color=INK, ha="right", va="bottom")

    # the bar
    ax.add_patch(Rectangle((tl_l, tl_y), tl_r - tl_l, tl_h, facecolor=L2,
                           edgecolor=GRAY, lw=0.5, transform=ax.transAxes,
                           clip_on=False, zorder=1))
    for t in range(0, int(CLIP_DUR) + 1):
        ax.plot([tx(t), tx(t)], [tl_y - 0.007, tl_y], color=GRAY, lw=0.5,
                transform=ax.transAxes, clip_on=False)
        ax.text(tx(t), tl_y - 0.014, f"{t}", fontsize=6.5, color=GRAY,
                ha="center", va="top")
    ax.text(tl_r, tl_y - 0.036, "seconds", fontsize=6.5, color=GRAY,
            ha="right", va="top")

    # the two windows drawn on the bar, with numeric callouts (serif)
    for (name, a, b), col, lab in ((EV_A, RED, "A"), (EV_B, INK, "B")):
        w = max(tx(b) - tx(a), 0.004)
        ax.add_patch(Rectangle((tx(a), tl_y), w, tl_h, facecolor=col, alpha=0.30,
                               edgecolor=col, lw=0.8, transform=ax.transAxes,
                               clip_on=False, zorder=3))
    # A window is only 0.05 s wide: label it off to the side, B centred
    ax.text(tx(EV_A[1]) - 0.006, tl_y + tl_h + 0.006,
            f"A  [{EV_A[1]:.2f}, {EV_A[2]:.2f}] s", fontsize=6.5, color=RED,
            ha="right", va="bottom")
    ax.text(tx((EV_B[1] + EV_B[2]) / 2), tl_y + tl_h + 0.006,
            f"B  [{EV_B[1]:.2f}, {EV_B[2]:.2f}] s", fontsize=6.5, color=INK,
            ha="center", va="bottom")

    # ===================================================================== #
    # 3. the real frames: an even filmstrip just below the timeline, joined to
    #    each named instant by a short elbow leader (little empty space between).
    # ===================================================================== #
    strip_l, strip_r = 0.060, 0.965
    n = len(FRAME_TIMES)
    fgap = 0.014
    fw = (strip_r - strip_l - (n - 1) * fgap) / n
    fh = fw * (frames[FRAME_TIMES[0]].shape[0] / frames[FRAME_TIMES[0]].shape[1]) * AR
    fy = 0.150                                          # filmstrip baseline
    ftop = fy + fh                                      # shared strip top
    # a per-frame timestamp caption sits just ABOVE the strip, on one shared line,
    # clear of the seconds tick row above it
    cap_y = ftop + 0.026
    for i, t in enumerate(FRAME_TIMES):
        img = frames[t]
        x0 = strip_l + i * (fw + fgap)
        axi = fig.add_axes([x0, fy, fw, fh])
        axi.imshow(img); axi.set_xticks([]); axi.set_yticks([])
        flag = t in FRAME_NOTE
        for sp in axi.spines.values():
            sp.set_color(RED if flag else GRAY)
            sp.set_linewidth(1.0 if flag else HAIR)
        cx_frame = x0 + fw / 2
        cx_time = tx(t)
        ax.text(cx_frame, cap_y, f"{t:.2f} s" + (f"  {FRAME_NOTE[t]} starts" if flag else ""),
                fontsize=6.4, color=(RED if flag else GRAY), ha="center", va="bottom")
        # short elbow leader: from the caption up to a rail under the bar, then the tick
        y_rail = tl_y - 0.020
        elbow_leader(ax, cx_frame, cap_y + 0.016, cx_time, y_rail, tl_y,
                     color=(RED if flag else L1), lw=(0.6 if flag else 0.5))
        ax.plot([cx_time, cx_time], [tl_y, tl_y - 0.006],
                color=(RED if flag else GRAY), lw=0.7,
                transform=ax.transAxes, clip_on=False, zorder=4)

    # ===================================================================== #
    # 4. the arithmetic the compiler applied (compact footer strip, serif)
    # ===================================================================== #
    ax.text(0.014, 0.070,
            r"How the order was decided:  the slack is $\delta = \frac{1}{4}$ of the "
            rf"shorter window = {TOL:.4f} s, and A is before B iff  "
            r"$A_{\mathrm{end}} \leq B_{\mathrm{start}} + \delta$.",
            fontsize=7.0, color=GRAY, va="bottom", ha="left")
    ax.text(0.014, 0.032,
            rf"${EV_A[2]:.2f} \leq {EV_B[1]:.2f} + {TOL:.4f}$", fontsize=7.4,
            color=INK, va="bottom", ha="left")
    ax.text(0.245, 0.032,
            r"$\rightarrow$  holds, so the claim is supported.  Otherwise the windows "
            "carry no order and it stays unresolved.",
            fontsize=7.0, color=GRAY, va="bottom", ha="left")

    for ext in ("pdf", "png"):
        out = HERE / f"fig_plan_chain.{ext}"
        fig.savefig(out, dpi=220 if ext == "png" else None,
                    bbox_inches=None,
                    metadata={"CreationDate": None} if ext == "pdf" else None)
        print(f"[write] {out.name}  {out.stat().st_size/1024:.0f} kB")
    plt.close(fig)


if __name__ == "__main__":
    main()
