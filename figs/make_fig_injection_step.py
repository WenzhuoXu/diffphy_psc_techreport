#!/usr/bin/env python3
"""Build fig_injection_step for the technical report.

The question the figure answers: the depth control does not have to be applied
for the whole denoising process; if it is released partway through, does the
generated motion still follow the simulation?

Two panels, one visual system:

  LEFT  -- the release-point sweep. One clip per release point, three scenes,
           scored twice: over the AIRBORNE window (up to the frame at which the
           simulated object first reaches the ground) and over the AFTER-CONTACT
           window (that frame to the end). x axis is the NOISE LEVEL at which the
           control was switched off; the step index of each run is printed
           underneath as a label only, because the same step index sits at a
           different noise level under a different sampling schedule.
  RIGHT -- the seeded measurement at one release point, 16 seeds per cell, with
           95% confidence intervals of the mean.

NOTHING here is read from a cached score file for the left panel: the control and
every generated clip are re-tracked from the mp4s and the correlations recomputed,
so the plotted curve is measured at figure-build time. The right panel's means and
intervals come from the seeded aggregate JSON and are re-derived from that file's
own per-clip rows before plotting, and the script aborts if the two disagree.

Everything printed to stdout is measured or read: the resolved font files, the
per-clip frame counts, the detected contact frame per scene, every plotted point
with its sample count, and the figure geometry.

Run:  python3 /Users/wenzhuox/diffphy_psc/techreport/figs/make_fig_injection_step.py

Idempotent: re-running overwrites the same two output files byte-identically.
"""
from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# paths (absolute; every source is read-only)
# --------------------------------------------------------------------------- #
OUT_DIR = Path("/Users/wenzhuox/diffphy_psc/techreport/figs")

# the release-point sweep: <scene>_control.mp4 + <scene>_{nodrop,drop8,...}.mp4
SWEEP = Path("/Users/wenzhuox/diffphy_exp028/artifacts/runs/exp028/inspect/sweep")

# the seeded replication at one release point
SEEDED = Path("/Users/wenzhuox/diffphy_exp031/artifacts/exp031b/rep_aggregate_FINAL.json")

# --------------------------------------------------------------------------- #
# report visual system -- this palette and nothing else
# --------------------------------------------------------------------------- #
ADOBE_RED = "#FA0F00"   # single sparing accent: the one curve that breaks
INK = "#1A1A1A"         # headings, primary marks
GRAY = "#6E6E6E"        # labels, every rule, the secondary marks
LIGHT = "#E5E5E5"       # the below-threshold band
FAINT = "#F5F5F5"       # reserved fill grey

HAIRLINE = 0.6          # pt, every rule and keyline

FONT_DIRS = [
    "/Users/wenzhuox/Library/Caches/Tectonic/bundles/data/*",  # the report's own OTFs
    str(Path.home() / "Library/Fonts"),
    "/Library/Fonts",
    "/Library/Application Support/Adobe/*/*/Fonts",
]
FONT_PATTERNS = ("SourceSansPro-*.otf", "SourceSerifPro-*.otf", "SourceCodePro-*.otf")

SANS_FAMILIES = ["Source Sans 3", "Source Sans Pro"]
SERIF_FAMILIES = ["Source Serif 4", "Source Serif Pro"]
MONO_FAMILIES = ["Source Code Pro"]


def apply_style() -> None:
    """Register the report's Source faces and declare them explicitly.

    Same routine as the other generation figures: the faces are declared by name
    with no generic fallback, the resolved font FILE for each family is printed,
    and a missing Source Sans is a hard error rather than a silent substitution.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm

    found: dict[str, str] = {}
    for root in FONT_DIRS:
        for base in glob.glob(root):
            for pat in FONT_PATTERNS:
                for path in glob.glob(str(Path(base) / pat)):
                    found.setdefault(Path(path).name, path)
    for path in found.values():
        try:
            fm.fontManager.addfont(path)
        except Exception:  # a font that will not parse is not fatal
            pass

    families = {f.name for f in fm.fontManager.ttflist}
    print(f"[fonts] registered {len(found)} Source font files")
    print(f"[fonts] Source families now available: "
          f"{sorted(n for n in families if n.startswith('Source'))}")
    if not any(n in families for n in SANS_FAMILIES):
        raise SystemExit(
            "Source Sans not found and no fallback is acceptable for this report. "
            "Searched: " + ", ".join(FONT_DIRS)
        )

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": SANS_FAMILIES,
        "font.serif": SERIF_FAMILIES,
        "font.monospace": MONO_FAMILIES,
        "font.size": 8.0,
        "text.color": INK,
        "axes.edgecolor": GRAY,
        "axes.linewidth": HAIRLINE,
        "axes.grid": False,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    for label, fams in (("sans-serif", SANS_FAMILIES),
                        ("serif", SERIF_FAMILIES),
                        ("monospace", MONO_FAMILIES)):
        prop = fm.FontProperties(family=fams)
        path = fm.findfont(prop, fallback_to_default=False)
        print(f"[fonts] {label:<10s} declared {fams} -> resolved "
              f"{Path(path).name}  ({fm.FontProperties(fname=path).get_name()})")


def semibold(size: float):
    """Source Sans Semibold, the report's heading weight."""
    from matplotlib import font_manager as fm
    for root in FONT_DIRS:
        for base in glob.glob(root):
            hits = glob.glob(str(Path(base) / "SourceSansPro-Semibold.otf"))
            if hits:
                return fm.FontProperties(fname=sorted(hits)[0], size=size)
    return fm.FontProperties(family=SANS_FAMILIES, size=size, weight="semibold")


# --------------------------------------------------------------------------- #
# the sampling schedule: step index -> noise level
#
# Ported from the schedule the runs actually used
# (scripts/exp028/bernini_schedule_patch.py::unipc_flow_timesteps): a flow-matching
# UniPC schedule at a given step count and schedule shape. The point of having it
# here is that the figure's x axis is the NOISE LEVEL, derived from the step index,
# not the step index itself.
# --------------------------------------------------------------------------- #
def schedule_sigmas(n_steps: int, shape: float, n_train: int = 1000) -> np.ndarray:
    alphas = np.linspace(1.0, 1.0 / n_train, n_steps + 1)
    sig = 1.0 - alphas
    sig = np.flip(shape * sig / (1 + (shape - 1) * sig))[:-1]
    return sig  # already in [0,1]: timesteps/n_train


SWEEP_STEPS = 40      # the sweep and the seeded run both used 40 denoising steps
SWEEP_SHAPE = 5.0     # and this schedule shape


# --------------------------------------------------------------------------- #
# tracking (the same colour-blob tracker the runs were scored with)
# --------------------------------------------------------------------------- #
# HSV key per scene, matching the object the simulation actually renders.
HSV = {
    "projectile_55": [((18, 90, 110), (38, 255, 255))],                        # yellow ball
    "ball_bounce": [((0, 110, 60), (10, 255, 255)),
                    ((170, 110, 60), (179, 255, 255))],                        # red (hue wraps)
    "ramp_ball": [((95, 80, 50), (135, 255, 255))],                            # blue ball
}
MIN_AREA = 60   # pixel-area gate: rejects specks too small to be the object


def track(path: Path, ranges, min_area: int = MIN_AREA):
    """Largest colour blob per frame, in normalised image coords, y measured up.

    NaN on frames where no blob above the area gate is found (the object has left
    the frame, or is not resolvable).
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    xs, ys = [], []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        h, w = fr.shape[:2]
        hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        mask = np.zeros((h, w), np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = cv2.medianBlur(mask, 5)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) >= min_area:
                m = cv2.moments(c)
                if m["m00"] > 0:
                    xs.append(m["m10"] / m["m00"] / w)
                    ys.append(1.0 - m["m01"] / m["m00"] / h)
                    continue
        xs.append(np.nan)
        ys.append(np.nan)
    cap.release()
    return np.array(xs, float), np.array(ys, float)


def contact_frame(y: np.ndarray, floor_frac: float = 0.10) -> int:
    """First frame at/after the object's highest point at which it is back on the floor.

    Read off the CONTROL's own vertical path, so the airborne/after-contact split
    is derived per scene from the simulation rather than hand-set.
    """
    valid = np.flatnonzero(~np.isnan(y))
    if valid.size < 3:
        return len(y)
    yv = y[valid]
    lo, hi = float(yv.min()), float(yv.max())
    if hi - lo < 1e-6:
        return len(y)
    peak = int(valid[int(np.argmax(yv))])
    thresh = lo + floor_frac * (hi - lo)
    for i in valid:
        if i >= peak and y[i] <= thresh:
            return int(i)
    return len(y)


def agreement(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray,
              sl: slice, min_n: int = 8):
    """Correlation of generated vs control path over commonly-tracked frames in `sl`.

    Returns (n_common, r_horizontal, r_vertical); r is None when the window has
    too few common frames or one series does not vary.
    """
    A, B, C, D = a[sl], b[sl], c[sl], d[sl]
    m = ~(np.isnan(A) | np.isnan(B) | np.isnan(C) | np.isnan(D))
    n = int(m.sum())
    if n < min_n:
        return n, None, None

    def r(u, v):
        if np.std(u) < 1e-12 or np.std(v) < 1e-12:
            return None
        return float(np.corrcoef(u, v)[0, 1])
    return n, r(A[m], C[m]), r(B[m], D[m])


# --------------------------------------------------------------------------- #
# measure panel 1: the release-point sweep, re-tracked from the mp4s
# --------------------------------------------------------------------------- #
SWEEP_SCENES = [
    ("projectile_55", "Ball thrown across a field"),
    ("ball_bounce", "Ball dropped onto a floor"),
    ("ramp_ball", "Ball rolling off a ramp"),
]
SWEEP_ARMS = ["nodrop", "drop8", "drop14", "drop20", "drop26", "drop32"]


def measure_sweep() -> dict:
    sig = schedule_sigmas(SWEEP_STEPS, SWEEP_SHAPE)
    print(f"\n[sweep] schedule: {SWEEP_STEPS} denoising steps, schedule shape "
          f"{SWEEP_SHAPE} -> noise level per release step:")
    for arm in SWEEP_ARMS[1:]:
        k = int(arm[4:])
        print(f"          step {k:2d} -> sigma {sig[k]:.4f}")
    # the point of reporting sigma rather than the step index, shown numerically
    alt = schedule_sigmas(SWEEP_STEPS, 3.0)
    print(f"[sweep] the step index is schedule-dependent: step 8 sits at sigma "
          f"{sig[8]:.4f} on this schedule and {alt[8]:.4f} on a shape-3.0 schedule; "
          f"sigma {sig[8]:.4f} lands nearest step "
          f"{int(np.argmin(abs(alt - sig[8])))} there.")

    out = {}
    for scene, title in SWEEP_SCENES:
        cx, cy = track(SWEEP / f"{scene}_control.mp4", HSV[scene])
        land = contact_frame(cy)
        print(f"\n[sweep] {scene}: control {len(cx)} frames, "
              f"{int(np.isfinite(cx).sum())} tracked, "
              f"first ground contact at frame {land} (from the control's own path)")
        rows = []
        for arm in SWEEP_ARMS:
            ox, oy = track(SWEEP / f"{scene}_{arm}.mp4", HSV[scene])
            n = min(len(cx), len(ox))
            a, b, c, d = cx[:n], cy[:n], ox[:n], oy[:n]
            L = min(land, n)
            n_air, air_x, air_y = agreement(a, b, c, d, slice(0, L))
            n_post, post_x, post_y = agreement(a, b, c, d, slice(L, None))
            n_all, all_x, all_y = agreement(a, b, c, d, slice(None))
            k = None if arm == "nodrop" else int(arm[4:])
            s = None if k is None else float(sig[k])
            rows.append(dict(arm=arm, step=k, sigma=s,
                             n_air=n_air, air_x=air_x, air_y=air_y,
                             n_post=n_post, post_x=post_x, post_y=post_y,
                             n_all=n_all, all_x=all_x, all_y=all_y,
                             cov=round(n_all / n, 3)))
            f = lambda v: "  n/a" if v is None else f"{v:+.3f}"     # noqa: E731
            print(f"          {arm:7s} step={str(k):>4s} sigma={'  none' if s is None else f'{s:.4f}'}"
                  f" | airborne n={n_air:2d} horiz={f(air_x)} vert={f(air_y)}"
                  f" | after-contact n={n_post:2d} horiz={f(post_x)} vert={f(post_y)}"
                  f" | whole n={n_all:2d} coverage={n_all / n:.3f}")
        out[scene] = dict(title=title, land=land, rows=rows)
    return out


# --------------------------------------------------------------------------- #
# measure panel 2: the seeded run at one release point
# --------------------------------------------------------------------------- #
# Cell keys in the aggregate file, and what each cell is in plain terms. The two
# release variants differ only in whether the sampler's internal history is also
# reset at the release; both release the control at the same noise level.
SEED_CELLS = [
    ("a4", "control never released"),
    ("a1", "released, history reset"),
    ("a2", "released, no reset"),
]
SEED_SCENES = [("projectile_55", "Ball thrown across a field"),
               ("ball_bounce", "Ball dropped onto a floor")]


def _mean_ci(vals):
    """mean and 95% t-interval over the non-missing values (the aggregate's own rule)."""
    from scipy import stats
    v = [float(x) for x in vals if x is not None]
    n = len(v)
    if n == 0:
        return 0, None, None
    a = np.asarray(v, float)
    m = float(a.mean())
    if n < 2:
        return n, round(m, 4), None
    se = a.std(ddof=1) / math.sqrt(n)
    half = float(stats.t.ppf(0.975, n - 1)) * se
    return n, round(m, 4), [round(m - half, 4), round(m + half, 4)]


def measure_seeded() -> dict:
    d = json.loads(SEEDED.read_text())
    cfg, cells, per_clip = d["config"], d["cells"], d["per_clip"]
    break_line = float(cfg["collapse_rx"])
    print(f"\n[seeded] {cfg['n_clips']} clips, area gate {cfg['min_area']} px, "
          f"break line {break_line}, phantom rule whole<{cfg['phantom_full_rx']} "
          f"and airborne>{cfg['phantom_arc_rx']}")

    # the one release point this run used, read from the run's own per-clip meta
    meta = Path("/Users/wenzhuox/diffphy_exp031/artifacts/runs/exp031/"
                "e31cases_rep/meta.jsonl")
    sigmas = {round(float(r["drop_sigma"]), 4)
              for r in (json.loads(l) for l in meta.read_text().splitlines() if l.strip())
              if r.get("drop_sigma") is not None}
    steps = {int(r["drop_index"])
             for r in (json.loads(l) for l in meta.read_text().splitlines() if l.strip())
             if r.get("drop_index") is not None}
    if len(sigmas) != 1 or len(steps) != 1:
        raise SystemExit(f"expected one release point, found sigmas={sigmas} steps={steps}")
    sigma = sigmas.pop()
    step = steps.pop()
    # cross-check against the schedule recomputed here
    recomputed = float(schedule_sigmas(SWEEP_STEPS, SWEEP_SHAPE)[step])
    print(f"[seeded] release point: step {step} of {SWEEP_STEPS} = noise level "
          f"sigma {sigma:.4f} (schedule recomputed here gives {recomputed:.4f})")
    if abs(recomputed - sigma) > 5e-4:
        raise SystemExit("release noise level in the run meta disagrees with the schedule")

    out = {"break_line": break_line, "sigma": sigma, "step": step,
           "n_clips": int(cfg["n_clips"]), "scenes": {}}
    for scene, title in SEED_SCENES:
        cell_out = []
        for arm, label in SEED_CELLS:
            cell = cells[f"{scene}|{arm}"]
            rows = [r for r in per_clip if r["scene"] == scene and r["arm"] == arm]
            entry = dict(arm=arm, label=label, n_seeds=int(cell["n_seeds"]),
                         land=int(cell["land_frames"][0]))
            for win in ("arc", "full"):
                for ax in ("rx", "ry"):
                    stored = cell[win][ax]
                    n, m, ci = _mean_ci([r[win][ax] for r in rows])
                    # the figure must not drift from the file it claims to plot
                    if n != stored["n"] or abs(m - stored["mean"]) > 1e-4:
                        raise SystemExit(
                            f"recomputed {scene}|{arm} {win}.{ax} = {n},{m} but file "
                            f"says {stored['n']},{stored['mean']}")
                    entry[f"{win}_{ax}"] = dict(n=n, mean=m, ci=ci)
                below = sum(1 for r in rows
                            if r[win]["rx"] is not None and r[win]["rx"] < break_line)
                stored_rate = cell[win]["collapse_rate"]
                if abs(stored_rate * len(rows) - below) > 1e-9:
                    raise SystemExit(f"collapse count mismatch on {scene}|{arm} {win}")
                entry[f"{win}_collapse"] = (below, len(rows))
            entry["phantom"] = (int(round(cell["phantom_rate"] * cell["phantom_n"])),
                                int(cell["phantom_n"]))
            cell_out.append(entry)
            print(f"[seeded] {scene:14s} {arm} ({label}): n={entry['n_seeds']} seeds, "
                  f"contact frame {entry['land']}")
            for win, wl in (("arc", "airborne"), ("full", "whole clip")):
                for ax, al in (("rx", "horizontal"), ("ry", "vertical")):
                    e = entry[f"{win}_{ax}"]
                    ci = "none" if e["ci"] is None else f"[{e['ci'][0]:+.3f},{e['ci'][1]:+.3f}]"
                    print(f"            {wl:10s} {al:10s} mean={e['mean']:+.4f} "
                          f"95% CI {ci} n={e['n']}")
                print(f"            {wl:10s} below break line: "
                      f"{entry[f'{win}_collapse'][0]}/{entry[f'{win}_collapse'][1]}")
            print(f"            flight-clean-but-ending-broken: "
                  f"{entry['phantom'][0]}/{entry['phantom'][1]}")
        out["scenes"][scene] = dict(title=title, cells=cell_out)
    return out


# --------------------------------------------------------------------------- #
# layout -- same geometry family as the other generation figures
# --------------------------------------------------------------------------- #
FIG_W = 6.5           # inches, the report text width
PAD_IN = 0.06         # outer margin, identical on all four sides
CONTENT_W = FIG_W - 2 * PAD_IN

HEAD_H = 0.235        # heading band: title baseline + hairline rule
HEAD_BASE = 0.105
HEAD_RULE = 0.160
SUBHEAD_H = 0.185     # scene name above its own plot, so it never sits on the data
PANEL_GAP = 0.40      # between the two panels, horizontally
BLOCK_GAP = 0.26      # between stacked scene blocks
AX_H = 0.86           # plot height for one left-panel scene row
XLAB_H = 0.50         # under the last left plot: tick labels + step labels + axis title
RXLAB_H = 0.44        # under the last right plot: two-line group labels
LEGEND_H = 0.34       # under those: the two keys (each two lines tall)

# Y-axis tick labels and the axis title live in a GUTTER inside the content box.
# Without it the labels are drawn outside the figure canvas and get cut off, since
# the figure is deliberately saved without a tight bounding box.
GUTTER = 0.52         # left of every plot: axis title + tick labels

HEAD_FS = 8.8
LABEL_FS = 8.0
TICK_FS = 8.0
KEY_FS = 8.0

LEFT_W = CONTENT_W - 2 * GUTTER   # sweep panel: content width less its gutter
RIGHT_W = CONTENT_W - 2 * GUTTER  # seeded panel: same, stacked below


class Layout:
    """Place things in inches from the content box's top-left corner.

    The content box is inset by PAD_IN from every figure edge and the figure is
    saved without a tight bounding box, so the outer margin is exactly PAD_IN on
    all four sides rather than whatever a tight-bbox pass happens to leave.
    """

    def __init__(self, fig, w_in: float, h_in: float):
        self.fig, self.w, self.h = fig, w_in, h_in

    def _fx(self, x_in: float) -> float:
        return (PAD_IN + x_in) / self.w

    def _fy(self, y_in: float) -> float:
        return 1 - (PAD_IN + y_in) / self.h

    def rect(self, x_in, y_in, w_in, h_in):
        return [self._fx(x_in), self._fy(y_in + h_in), w_in / self.w, h_in / self.h]

    def text(self, x_in, y_in, s, **kw):
        return self.fig.text(self._fx(x_in), self._fy(y_in), s, **kw)

    def hrule(self, x0_in, x1_in, y_in, color, lw=HAIRLINE):
        from matplotlib.lines import Line2D
        y = self._fy(y_in)
        self.fig.add_artist(Line2D([self._fx(x0_in), self._fx(x1_in)], [y, y],
                                   transform=self.fig.transFigure, color=color,
                                   linewidth=lw, solid_capstyle="butt"))


def despine(ax, keep=("left", "bottom")):
    for name, sp in ax.spines.items():
        sp.set_visible(name in keep)
        sp.set_color(GRAY)
        sp.set_linewidth(HAIRLINE)
    ax.tick_params(length=2.4, width=HAIRLINE, colors=GRAY, labelcolor=INK,
                   labelsize=TICK_FS)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------- #
# draw
# --------------------------------------------------------------------------- #
def draw(sweep: dict, seeded: dict) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    n_rows = len(SWEEP_SCENES)
    n_sc_pre = len(SEED_SCENES)
    SEED_AX_H = 1.05                      # taller: its own tick labels need room
    sweep_h = HEAD_H + n_rows * AX_H + (n_rows - 1) * BLOCK_GAP + XLAB_H + LEGEND_H
    SEED_TOP = sweep_h + 0.30             # the stacked panel starts below the first
    seed_h = HEAD_H + n_sc_pre * SEED_AX_H + (n_sc_pre - 1) * BLOCK_GAP + 0.52 + LEGEND_H
    content_h = SEED_TOP + seed_h
    fig_h = content_h + 2 * PAD_IN
    fig = plt.figure(figsize=(FIG_W, fig_h))
    L = Layout(fig, FIG_W, fig_h)

    break_line = seeded["break_line"]

    # ---------------- left panel: the release-point sweep -------------------
    L.text(0.0, HEAD_BASE, "Retention against the release point",
           ha="left", va="baseline", fontproperties=semibold(HEAD_FS), color=INK)
    L.hrule(0.0, LEFT_W, HEAD_RULE, GRAY)

    # x axis: noise level, high noise on the LEFT (denoising runs left to right)
    sig_all = sorted({r["sigma"] for s in sweep.values() for r in s["rows"]
                      if r["sigma"] is not None}, reverse=True)
    x_lo, x_hi = min(sig_all) - 0.035, max(sig_all) + 0.035

    for i, (scene, _t) in enumerate(SWEEP_SCENES):
        info = sweep[scene]
        rows = [r for r in info["rows"] if r["sigma"] is not None]
        base = next(r for r in info["rows"] if r["sigma"] is None)
        y_top = HEAD_H + i * (AX_H + BLOCK_GAP)
        ax = fig.add_axes(L.rect(GUTTER, y_top, LEFT_W, AX_H))

        # the region a curve must stay out of: below the pre-registered break line
        ax.axhspan(-1.05, break_line, color=LIGHT, lw=0, zorder=0)

        # the never-released baseline, as a reference level
        for key, style in (("air_x", (0, (1.2, 1.4))), ("post_x", (0, (1.2, 1.4)))):
            pass  # baselines drawn once below, from the whole-clip windows

        series = [
            ("air_x", "airborne, across frame", GRAY, "o", "-", 1.0),
            ("air_y", "airborne, up/down", GRAY, "^", (0, (2.6, 1.6)), 1.0),
            ("post_x", "after contact, across frame", ADOBE_RED, "s", "-", 1.25),
            ("post_y", "after contact, up/down", INK, "D", (0, (2.6, 1.6)), 1.0),
        ]
        for key, _lab, color, marker, ls, lw in series:
            pts = [(r["sigma"], r[key]) for r in rows if r[key] is not None]
            if len(pts) < 2:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, ls=ls, lw=lw, color=color, marker=marker, ms=2.9,
                    mfc="white" if marker in ("^", "D") else color,
                    mec=color, mew=HAIRLINE, clip_on=False, zorder=3)

        # baseline (control never released) as a hairline reference for both windows
        for key, color in (("air_x", GRAY), ("post_x", ADOBE_RED)):
            v = base[key]
            if v is None:
                continue
            ax.plot([x_lo, x_hi], [v, v], color=color, lw=HAIRLINE,
                    ls=(0, (0.9, 1.6)), zorder=1)

        ax.set_xlim(x_hi, x_lo)   # high noise on the left
        ax.set_ylim(-1.05, 1.06)
        ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax.set_yticklabels(["−1.0", "−0.5", "0", "0.5", "1.0"])
        despine(ax)
        ax.set_ylabel("retention", fontsize=LABEL_FS, color=GRAY, labelpad=2.0)

        # scene name inside the axes, top-left, so it needs no extra band
        ax.text(0.012, 0.055, info["title"], transform=ax.transAxes,
                ha="left", va="bottom", fontproperties=semibold(LABEL_FS), color=INK)

        if i < n_rows - 1:
            ax.set_xticklabels([])
            ax.set_xticks(sig_all)
        else:
            ax.set_xticks(sig_all)
            ax.set_xticklabels([f"{s:.2f}" for s in sig_all])
            ax.set_xlabel("noise level at which the control was released",
                          fontsize=LABEL_FS, color=GRAY, labelpad=11.0)
            # the step index of each run, as a label only
            for s in sig_all:
                k = next(r["step"] for r in rows if r["sigma"] == s)
                ax.annotate(f"step {k}", xy=(s, 0), xycoords=("data", "axes fraction"),
                            xytext=(0, -15.5), textcoords="offset points",
                            ha="center", va="top", fontsize=TICK_FS, color=GRAY,
                            annotation_clip=False)

        # mark, on the scene that breaks, where retention collapses
        if scene == "projectile_55":
            bad = [r for r in rows if r["post_x"] is not None and r["post_x"] < break_line]
            if bad:
                w = max(bad, key=lambda r: r["sigma"])
                ax.annotate("runs opposite to\nthe simulation",
                            xy=(w["sigma"], w["post_x"]),
                            xytext=(w["sigma"] - 0.055, w["post_x"] - 0.30),
                            ha="left", va="top", fontsize=KEY_FS, color=ADOBE_RED,
                            linespacing=1.25,
                            arrowprops=dict(arrowstyle="-", color=ADOBE_RED,
                                            lw=HAIRLINE, shrinkA=1.0, shrinkB=2.0))

    # key, under the axis labels
    y_key = HEAD_H + n_rows * AX_H + (n_rows - 1) * BLOCK_GAP + XLAB_H + 0.055
    handles = [
        Line2D([], [], color=GRAY, lw=1.0, ls="-", marker="o", ms=2.9,
               mfc=GRAY, mec=GRAY, mew=HAIRLINE, label="airborne, across frame"),
        Line2D([], [], color=GRAY, lw=1.0, ls=(0, (2.6, 1.6)), marker="^", ms=2.9,
               mfc="white", mec=GRAY, mew=HAIRLINE, label="airborne, up/down"),
        Line2D([], [], color=ADOBE_RED, lw=1.25, ls="-", marker="s", ms=2.9,
               mfc=ADOBE_RED, mec=ADOBE_RED, mew=HAIRLINE,
               label="after contact, across frame"),
        Line2D([], [], color=INK, lw=1.0, ls=(0, (2.6, 1.6)), marker="D", ms=2.9,
               mfc="white", mec=INK, mew=HAIRLINE, label="after contact, up/down"),
        Line2D([], [], color=GRAY, lw=HAIRLINE, ls=(0, (0.9, 1.6)),
               label="control never released"),
    ]
    leg = fig.legend(handles=handles, loc="upper left", frameon=False,
                     bbox_to_anchor=(L._fx(0.0), L._fy(y_key)),
                     ncol=3, columnspacing=1.05, handlelength=2.1,
                     handletextpad=0.45, labelspacing=0.30, fontsize=KEY_FS)
    for t in leg.get_texts():
        t.set_color(GRAY)

    # ---------------- right panel: the seeded measurement -------------------
    rx = 0.0
    L.text(rx, SEED_TOP + HEAD_BASE,
           f"At one release point (noise level {seeded['sigma']:.2f}), "
           f"{seeded['scenes']['projectile_55']['cells'][0]['n_seeds']} seeds per bar",
           ha="left", va="baseline", fontproperties=semibold(HEAD_FS), color=INK)
    L.hrule(rx, RIGHT_W, SEED_TOP + HEAD_RULE, GRAY)

    # one small axes per scene, bars = the three cells, two windows side by side
    n_sc = len(SEED_SCENES)
    sc_h = SEED_AX_H
    for j, (scene, _t) in enumerate(SEED_SCENES):
        info = seeded["scenes"][scene]
        y_top = SEED_TOP + HEAD_H + j * (sc_h + BLOCK_GAP)
        ax = fig.add_axes(L.rect(rx + GUTTER, y_top, RIGHT_W, sc_h))
        ax.axhspan(-0.05, break_line, color=LIGHT, lw=0, zorder=0)

        groups = [("arc_rx", "airborne\nacross"), ("arc_ry", "airborne\nup/down"),
                  ("full_rx", "whole clip\nacross"), ("full_ry", "whole clip\nup/down")]
        n_g, n_c = len(groups), len(info["cells"])
        bw = 0.78 / n_c
        colors = {"a4": LIGHT, "a1": ADOBE_RED, "a2": GRAY}
        for ci, cell in enumerate(info["cells"]):
            xs = [gi + (ci - (n_c - 1) / 2) * bw for gi in range(n_g)]
            ys = [cell[k]["mean"] for k, _ in groups]
            los = [cell[k]["mean"] - cell[k]["ci"][0] for k, _ in groups]
            his = [cell[k]["ci"][1] - cell[k]["mean"] for k, _ in groups]
            ax.bar(xs, ys, width=bw * 0.92, color=colors[cell["arm"]],
                   edgecolor=GRAY if cell["arm"] == "a4" else colors[cell["arm"]],
                   linewidth=HAIRLINE, zorder=2)
            ax.errorbar(xs, ys, yerr=[los, his], fmt="none", ecolor=INK,
                        elinewidth=HAIRLINE, capsize=1.7, capthick=HAIRLINE, zorder=4)

        ax.set_xlim(-0.6, n_g - 0.4)
        ax.set_ylim(-0.05, 1.10)
        ax.set_yticks([0.0, 0.5, break_line, 1.0])
        ax.set_yticklabels(["0", "0.5", f"{break_line:.1f}", "1.0"])
        ax.set_xticks(range(n_g))
        ax.set_xticklabels([lab for _, lab in groups] if j == n_sc - 1 else [""] * n_g, linespacing=1.22)
        despine(ax)
        ax.set_ylabel("retention", fontsize=LABEL_FS, color=GRAY, labelpad=2.0)
        ax.text(0.012, 1.075, info["title"], transform=ax.transAxes,
                ha="left", va="bottom", fontproperties=semibold(LABEL_FS), color=INK)

    # key for the right panel, aligned with the left panel's key
    from matplotlib.patches import Patch
    rhandles = [
        Patch(facecolor=LIGHT, edgecolor=GRAY, lw=HAIRLINE, label="never released"),
        Patch(facecolor=ADOBE_RED, edgecolor=ADOBE_RED, lw=HAIRLINE,
              label="released, history reset"),
        Patch(facecolor=GRAY, edgecolor=GRAY, lw=HAIRLINE, label="released, no reset"),
    ]
    rleg = fig.legend(handles=rhandles, loc="upper left", frameon=False,
                      bbox_to_anchor=(L._fx(rx), L._fy(SEED_TOP + HEAD_H + n_sc * (sc_h + BLOCK_GAP) + 0.42)),
                      ncol=2, columnspacing=1.05, handlelength=1.25,
                      handletextpad=0.45, labelspacing=0.30, fontsize=KEY_FS)
    for t in rleg.get_texts():
        t.set_color(GRAY)

    print(f"\n[fig] figure {FIG_W:.2f} x {fig_h:.2f} in, content "
          f"{CONTENT_W:.2f} x {content_h:.2f} in, margin {PAD_IN:.2f} in on all four "
          f"sides, every rule {HAIRLINE:.1f} pt")
    print(f"[fig] left panel {LEFT_W:.2f} in wide ({n_rows} scene rows of "
          f"{AX_H:.2f} in), right panel {RIGHT_W:.2f} in wide "
          f"({n_sc} scene rows of {sc_h:.2f} in), panel gap {PANEL_GAP:.2f} in")
    print(f"[fig] every in-figure text size: heading {HEAD_FS} pt, labels "
          f"{LABEL_FS} pt, ticks {TICK_FS} pt, key {KEY_FS} pt (all >= 8 pt)")
    return save(fig, "fig_injection_step")


def save(fig, stem: str) -> list[Path]:
    """Write the PDF (for LaTeX) and a 200-dpi PNG (for review).

    Saved with the figure's full canvas (no tight bounding box) because the layout
    already reserves PAD_IN on all four sides. The PDF's creation date is pinned to
    None so re-running gives a byte-identical file.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext, kw in (("pdf", {"metadata": {"CreationDate": None}}), ("png", {"dpi": 200})):
        p = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(p, **kw)
        print(f"[write] {p}  ({p.stat().st_size / 1024:.0f} kB)")
        paths.append(p)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return paths


def main() -> int:
    apply_style()
    for p in (SWEEP, SEEDED):
        if not p.exists():
            raise SystemExit(f"missing source: {p}")
    sweep = measure_sweep()
    seeded = measure_seeded()
    draw(sweep, seeded)
    return 0


if __name__ == "__main__":
    sys.exit(main())
