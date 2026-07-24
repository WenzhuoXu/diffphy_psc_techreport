#!/usr/bin/env python3
"""Build the two generation figures for the technical report.

Both figures are made only of REAL frames decoded from the real artifacts, and
both are built from ONE shared block template so they read as a family:

    case heading + hairline rule
      simulated depth control  row of frames
      generated video          row of frames
      timestamps

  fig_generation_pipeline  -- one case (the bouncing ball).
  fig_generation_cases     -- two cases stacked (bouncing ball, ball thrown
                              across a field), each with its own timestamps.

Which frames are shown is COMPUTED, not chosen by hand: the ball is tracked in
every frame of each control render, the turning points of its vertical path are
detected, and the columns are the launch frame plus those turning points plus
fills placed where the ball moved the most. That guarantees every column is
motion-bearing (a real apex / contact / rebound) rather than a run of frames in
which the object sits still.

Everything printed to stdout is measured from the files: frame counts, frame
rate, duration, the resolved font files, the per-frame ball track, the chosen
frames with the ball's height above the floor, the ball-vs-background contrast
at each chosen frame, and a check that each generated clip really is the one
produced from its control render.

Run:  python3 /Users/wenzhuox/diffphy_psc/techreport/figs/make_fig_generation.py

Idempotent: re-running overwrites the same four output files.
"""
from __future__ import annotations

import glob
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# paths (absolute; sources are read-only)
# --------------------------------------------------------------------------- #
OUT_DIR = Path("/Users/wenzhuox/diffphy_psc/techreport/figs")
VACE = Path("/Users/wenzhuox/diffphy_vace/artifacts/runs/exp029")
CONTROLS = VACE / "controls"
GENERATED = VACE / "inspect/outputs"

# --------------------------------------------------------------------------- #
# report visual system -- this palette and nothing else
# --------------------------------------------------------------------------- #
# The report's palette. These two figures are frames plus type, so the only
# colours they draw are the accent, the text ink and the hairline grey; the two
# report fill greys (#E5E5E5, #F5F5F5) are for charts and are deliberately unused
# here, and nothing outside this list is drawn.
ADOBE_RED = "#FA0F00"   # single accent: the marker on the output row only
INK = "#1A1A1A"         # case headings
GRAY = "#6E6E6E"        # labels, timestamps, every rule and keyline

HAIRLINE = 0.6          # pt, every rule and keyline in both figures

FONT_DIRS = [
    "/Users/wenzhuox/Library/Caches/Tectonic/bundles/data/*",  # the report's own OTFs
    str(Path.home() / "Library/Fonts"),
    "/Library/Fonts",
    "/Library/Application Support/Adobe/*/*/Fonts",
]
FONT_PATTERNS = ("SourceSans*.otf", "SourceSerif*.otf", "SourceCode*.otf")

SANS_FAMILIES = ["Source Sans 3", "Source Sans Pro"]
SERIF_FAMILIES = ["Source Serif 4", "Source Serif Pro"]
MONO_FAMILIES = ["Source Code Pro"]


def apply_style() -> None:
    """Register the report's Source faces and declare them explicitly.

    The report is typeset in Source Sans / Source Serif / Source Code Pro. The
    faces are declared by name in rcParams with no generic fallback, the
    resolved font FILE for each family is printed, and a missing Source Sans is
    a hard error rather than a silent substitution.
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

    # prove which file each declared family actually resolves to
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
            hits = glob.glob(str(Path(base) / "SourceSans*Semibold.otf"))
            if hits:
                return fm.FontProperties(fname=sorted(hits)[0], size=size)
    return fm.FontProperties(family=SANS_FAMILIES, size=size, weight="semibold")


# --------------------------------------------------------------------------- #
# video reading / measurement
# --------------------------------------------------------------------------- #
@dataclass
class ClipInfo:
    path: Path
    n_frames: int
    fps: float
    duration_s: float
    width: int
    height: int


def probe(path: Path) -> ClipInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return ClipInfo(path, n, fps, n / fps if fps else float("nan"), w, h)


def read_all(path: Path) -> list[np.ndarray]:
    """Decode every frame sequentially (no seeking, so nothing can be skipped)."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if not frames:
        raise SystemExit(f"decoded zero frames from {path}")
    return frames


def _largest_blob(mask: np.ndarray) -> tuple | None:
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None
    j = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (float(cent[j][0]), float(cent[j][1]), float(stats[j, cv2.CC_STAT_AREA]),
            float(stats[j, cv2.CC_STAT_TOP] + stats[j, cv2.CC_STAT_HEIGHT]))


def silhouette_ball_track(frames: list[np.ndarray]) -> np.ndarray:
    """Track the ball in a silhouette render (white object on black).

    This is the authoritative track. The silhouette render is the same
    simulation as the depth render, so it gives the ball's exact footprint with
    no ambiguity about what is object and what is background.
    Returns (N,4): centre x, centre y, pixel area, bottom-most row. Frames where
    the ball is not present (it has left the frame) are -1.
    """
    out = np.full((len(frames), 4), -1.0)
    for i, f in enumerate(frames):
        blob = _largest_blob(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) > 127)
        if blob:
            out[i] = blob
    return out


def depth_ball_track(frames: list[np.ndarray], thresh: float = 8.0) -> np.ndarray:
    """Track the ball in a depth control render, as a cross-check only.

    The background of a depth render is fixed for the whole clip, so a
    per-pixel median over time is the background and the ball is whatever
    departs from it. (A per-row median does NOT work here: the projectile
    render's horizon is very slightly tilted, which leaves a fixed high-residual
    streak that the tracker latches onto once the real ball gets close to the
    floor.)
    """
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
    bg = np.median(np.stack(gray), axis=0)
    out = np.full((len(frames), 4), -1.0)
    for i, g in enumerate(gray):
        blob = _largest_blob(np.abs(g - bg) > thresh)
        if blob and blob[2] >= 60:
            out[i] = blob
    return out


def colored_ball_track(frames: list[np.ndarray], hue_lo: int, hue_hi: int,
                       s_min: int, v_min: int) -> np.ndarray:
    """Track the strongly coloured ball in a generated clip."""
    out = np.full((len(frames), 4), -1.0)
    for i, f in enumerate(frames):
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        hm = (h >= hue_lo) | (h <= hue_hi) if hue_lo > hue_hi else (h >= hue_lo) & (h <= hue_hi)
        blob = _largest_blob(hm & (s >= s_min) & (v >= v_min))
        if blob:
            out[i] = blob
    return out


def ball_contrast(depth_frame: np.ndarray, sil_frame: np.ndarray) -> tuple[float, float]:
    """Mean depth value inside the ball vs. in a ring just outside it.

    The silhouette render gives the exact ball footprint, so this measures how
    strongly the ball stands out from its local background in the depth render
    without touching the depth pixels.
    """
    m = cv2.cvtColor(sil_frame, cv2.COLOR_BGR2GRAY) > 127
    if not m.any():
        return float("nan"), float("nan")
    d = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ring = cv2.dilate(m.astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool) & (~m)
    return float(d[m].mean()), float(d[ring].mean())


# --------------------------------------------------------------------------- #
# choosing motion-bearing frames
# --------------------------------------------------------------------------- #
def turning_points(cy: np.ndarray, valid: np.ndarray, window: int = 3,
                   min_amp: float = 3.0) -> list[int]:
    """Frames where the ball's vertical direction reverses.

    A frame counts as a turning point when the ball is higher (or lower) than it
    was `window` frames earlier AND `window` frames later, by at least `min_amp`
    pixels on both sides. That picks out real apexes and real floor contacts and
    ignores the sub-pixel wobble of a ball that has come to rest.
    """
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return []
    lo, hi = int(idx.min()), int(idx.max())
    turns: list[int] = []
    for i in range(lo + window, hi - window + 1):
        a, b, c = cy[i - window], cy[i], cy[i + window]
        if not (valid[i - window] and valid[i] and valid[i + window]):
            continue
        is_max = b > a and b > c
        is_min = b < a and b < c
        if (is_max or is_min) and min(abs(b - a), abs(b - c)) >= min_amp:
            turns.append(i)
    # keep only the first of any cluster of adjacent turns
    kept: list[int] = []
    for i in turns:
        if not kept or i - kept[-1] > window:
            kept.append(i)
    return kept


def choose_columns(cy: np.ndarray, valid: np.ndarray, n_cols: int) -> tuple[list[int], list[int]]:
    """Pick `n_cols` frames that actually carry the motion.

    Starts from the launch frame plus every turning point of the vertical path,
    then repeatedly splits whichever remaining interval the ball travelled
    furthest through vertically. So the columns are always genuine events, and
    the filler frames land mid-flight rather than in a static stretch.
    Returns (chosen frames, the turning points that seeded them).
    """
    idx = np.flatnonzero(valid)
    first = int(idx.min())
    turns = turning_points(cy, valid)
    chosen = sorted({first, *turns})
    if len(chosen) > n_cols:
        chosen = chosen[:n_cols]
    while len(chosen) < n_cols:
        gaps = []
        for a, b in zip(chosen, chosen[1:]):
            if b - a >= 2:
                gaps.append((abs(cy[b] - cy[a]), a, b))
        if not gaps:
            break
        _, a, b = max(gaps)
        chosen.append(int(round((a + b) / 2)))
        chosen = sorted(set(chosen))
    return chosen, turns


# --------------------------------------------------------------------------- #
# ONE shared block template -- both figures are built from this
# --------------------------------------------------------------------------- #
FIG_W = 6.5          # inches, the report text width
PAD_IN = 0.06        # outer margin, identical on all four sides of both figures
CONTENT_W = FIG_W - 2 * PAD_IN
N_COLS = 5           # frames per row, identical in both figures
GUTTER = 0.92        # left label gutter
COL_GAP = 0.075      # between frame columns
ROW_GAP = 0.06       # between the control row and the generated row
BLOCK_GAP = 0.17     # between stacked cases
HEAD_H = 0.235       # heading band: title baseline + hairline rule
HEAD_BASE = 0.105    # title baseline inside the heading band
HEAD_RULE = 0.160    # hairline rule inside the heading band
TS_GAP = 0.085       # frames -> timestamps
TS_H = 0.105         # timestamp line; sized so the ink below the last timestamp
                     # clears the content edge by the same amount as the ink
                     # above the first heading, i.e. the figure looks equally
                     # padded top and bottom (verified from the rendered PNG)
ROW_RULE_LEN = 0.30  # short row-marker rule in the gutter
TICK_LEN = 0.05      # height tick, flush to the left edge of a control frame

HEAD_FS = 8.8
LABEL_FS = 8.0
TS_FS = 8.0

ROW_LABELS = ("Simulated\ndepth control", "Generated\nvideo")
ROW_RULE_COLORS = (GRAY, ADOBE_RED)   # the accent marks the output row


class Layout:
    """Place things in inches from the content box's top-left corner.

    The content box is inset by PAD_IN from every figure edge, and the figures
    are saved without a tight bounding box, so the outer margin is exactly
    PAD_IN on all four sides of both figures rather than whatever a tight-bbox
    pass happens to leave.
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


def col_w() -> float:
    return (CONTENT_W - GUTTER - COL_GAP * (N_COLS - 1)) / N_COLS


def col_x(c: int) -> float:
    return GUTTER + c * (col_w() + COL_GAP)


def block_h(ch: float) -> float:
    return HEAD_H + 2 * ch + ROW_GAP + TS_GAP + TS_H


def show_frame(ax, frame: np.ndarray) -> None:
    """A frame, unaltered, with a hairline keyline.

    The keyline is needed and is not a default border: the depth renders fade to
    near-white (253/255) along the bottom, so without it the panel edge would
    dissolve into the white page. Every panel in both figures gets the same
    hairline so the two rows sit on the same grid.
    """
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), interpolation="antialiased")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(GRAY)
        s.set_linewidth(HAIRLINE)


def draw_block(L: Layout, y_top: float, case: dict, ch: float) -> float:
    """Draw one complete case block. Returns the y of its bottom edge."""
    L.text(0.0, y_top + HEAD_BASE, case["title"], ha="left", va="baseline",
           fontproperties=semibold(HEAD_FS), color=INK)
    L.hrule(0.0, CONTENT_W, y_top + HEAD_RULE, GRAY)

    fps, idx = case["fps"], case["idx"]
    cw = col_w()
    for r, frames in enumerate((case["ctrl"], case["gen"])):
        y = y_top + HEAD_H + r * (ch + ROW_GAP)
        L.hrule(0.0, ROW_RULE_LEN, y + 0.045, ROW_RULE_COLORS[r])
        L.text(0.0, y + 0.175, ROW_LABELS[r], ha="left", va="top",
               fontsize=LABEL_FS, color=GRAY, linespacing=1.30)
        for c, i in enumerate(idx):
            x = col_x(c)
            ax = L.fig.add_axes(L.rect(x, y, cw, ch))
            show_frame(ax, frames[i])
            if r == 0:
                # Measured height marker for the control frame: a hairline tick
                # OUTSIDE the panel, flush against its left edge, at the ball's
                # measured centre height. The frame pixels are never touched --
                # brightening the ball would misrepresent the control signal --
                # so this is what keeps the ball's height legible in the columns
                # where it sits against the pale end of the depth gradient. It is
                # the secondary grey, not the accent, because it is a scale mark
                # rather than something to look at.
                frac = case["cy"][i] / case["frame_h"]
                L.hrule(x - TICK_LEN, x, y + frac * ch, GRAY)

    y_ts = y_top + HEAD_H + 2 * ch + ROW_GAP + TS_GAP
    for c, i in enumerate(idx):
        L.text(col_x(c) + cw / 2, y_ts, f"t = {i / fps:.2f} s",
               ha="center", va="top", fontsize=TS_FS, color=GRAY)
    return y_top + block_h(ch)


def build_figure(cases: list[dict], aspect: float, stem: str) -> list[Path]:
    """Both figures come through here, so the template cannot drift apart."""
    import matplotlib.pyplot as plt

    ch = col_w() / aspect
    content_h = len(cases) * block_h(ch) + (len(cases) - 1) * BLOCK_GAP
    fig_h = content_h + 2 * PAD_IN
    fig = plt.figure(figsize=(FIG_W, fig_h))
    L = Layout(fig, FIG_W, fig_h)
    y = 0.0
    for case in cases:
        y = draw_block(L, y, case, ch) + BLOCK_GAP
    print(f"  [{stem}] {len(cases)} case block(s), {N_COLS} columns, "
          f"gutter {GUTTER:.3f} in, panel {col_w():.3f} x {ch:.3f} in, "
          f"column gap {COL_GAP:.3f} in, row gap {ROW_GAP:.3f} in, "
          f"case gap {BLOCK_GAP:.3f} in")
    print(f"  [{stem}] figure {FIG_W:.2f} x {fig_h:.2f} in, content "
          f"{CONTENT_W:.2f} x {content_h:.2f} in, margin {PAD_IN:.2f} in on all four sides, "
          f"every rule {HAIRLINE:.1f} pt")
    return save(fig, stem)


def save(fig, stem: str) -> list[Path]:
    """Write the PDF (for LaTeX) and a 200-dpi PNG (for review).

    Saved with the figure's full canvas (no tight bounding box), because the
    layout already reserves PAD_IN on all four sides; a tight pass would crop to
    the drawn ink and leave the four margins unequal.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext, kw in (("pdf", {"metadata": {"CreationDate": None}}), ("png", {"dpi": 200})):
        p = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(p, **kw)
        paths.append(p)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------- #
def main() -> int:
    apply_style()

    # Plain descriptive case titles, checked against what the frames show: the
    # first generated clip is a red ball dropping and bouncing on a studio
    # floor; the second is a yellow ball arcing over a grass field under sky.
    scenes = [
        {
            "title": "Bouncing ball",
            "ctrl": CONTROLS / "ball_bounce_depth.mp4",
            "sil": CONTROLS / "ball_bounce_sil.mp4",
            "gen": GENERATED / "w22_bb_depth.mp4",
            "hue": (170, 10, 90, 60),    # red ball; hue wraps 0 on OpenCV's 0-179 scale
        },
        {
            "title": "Ball thrown across a field",
            "ctrl": CONTROLS / "projectile_55_depth.mp4",
            "sil": CONTROLS / "projectile_55_sil.mp4",
            "gen": GENERATED / "w22_proj_depth.mp4",
            "hue": (18, 33, 200, 150),   # yellow ball
        },
    ]

    print("\n" + "=" * 78)
    print("SOURCE CLIPS")
    print("=" * 78)
    for sc in scenes:
        ci, gi = probe(sc["ctrl"]), probe(sc["gen"])
        for tag, info in (("control  ", ci), ("generated", gi)):
            print(f"  {sc['title']:<26s} {tag}  {info.path.name:<26s} "
                  f"{info.n_frames:3d} frames  {info.fps:.2f} fps  "
                  f"{info.duration_s:.3f} s  {info.width}x{info.height}")
        aligned = (ci.n_frames == gi.n_frames and abs(ci.fps - gi.fps) < 1e-6)
        print(f"  {sc['title']:<26s} same frame count and rate in both: {aligned}")
        if not aligned:
            raise SystemExit(f"{sc['title']}: control and generated clip do not line up")
        sc.update(ctrl_info=ci, gen_info=gi,
                  ctrl_frames=read_all(sc["ctrl"]), sil_frames=read_all(sc["sil"]),
                  gen_frames=read_all(sc["gen"]))
        print(f"  {sc['title']:<26s} decoded {len(sc['ctrl_frames'])} control, "
              f"{len(sc['sil_frames'])} silhouette, {len(sc['gen_frames'])} generated frames")

    print("\n" + "=" * 78)
    print("BALL TRACK IN EACH CONTROL RENDER  (measured on every frame)")
    print("=" * 78)
    for sc in scenes:
        sil = silhouette_ball_track(sc["sil_frames"])
        dep = depth_ball_track(sc["ctrl_frames"])
        valid = sil[:, 0] >= 0
        sc.update(track=sil, valid=valid)
        both = valid & (dep[:, 0] >= 0)
        dy = np.abs(sil[both, 1] - dep[both, 1])
        print(f"  {sc['title']}: ball found in {int(valid.sum())}/{len(valid)} frames "
              f"(it leaves the frame in the rest)")
        print(f"  {sc['title']}: cross-check -- the same simulation tracked again in the depth "
              f"render agrees to {dy.max():.1f} px worst case, {np.median(dy):.1f} px median, "
              f"over {int(both.sum())} frames; frames found by each: "
              f"{int(valid.sum())} vs {int((dep[:, 0] >= 0).sum())}")
        if dy.max() > 6.0:
            raise SystemExit(
                f"{sc['title']}: the two independent tracks of the same simulation disagree by "
                f"{dy.max():.1f} px -- refusing to pick frames from a track that may be wrong"
            )
        cy = sil[:, 1]
        vh = sc["ctrl_info"].height
        print(f"  {sc['title']}: highest point of the ball's centre y = {cy[valid].min():.1f} px, "
              f"lowest = {cy[valid].max():.1f} px  (frame is {vh} px tall)")

    print("\n" + "=" * 78)
    print("PAIRING CHECK  (does the generated motion follow the control's motion?)")
    print("=" * 78)
    for sc in scenes:
        gen = colored_ball_track(sc["gen_frames"], *sc["hue"])
        ctrl = sc["track"]
        usable = [i for i in range(min(len(ctrl), len(gen)))
                  if ctrl[i][0] >= 0 and ctrl[i][2] >= 40 and gen[i][0] >= 0 and gen[i][2] >= 40]
        cx = np.array([ctrl[i][0] for i in usable]); cyv = np.array([ctrl[i][1] for i in usable])
        gx = np.array([gen[i][0] for i in usable]); gyv = np.array([gen[i][1] for i in usable])
        print(f"  {sc['title']}: ball tracked in both clips on {len(usable)}/"
              f"{len(sc['ctrl_frames'])} frames (through frame {max(usable)})")
        print(f"  {sc['title']}: path correlation {np.corrcoef(cx, gx)[0, 1]:+.4f} across the "
              f"frame, {np.corrcoef(cyv, gyv)[0, 1]:+.4f} up and down")
        print(f"  {sc['title']}: median disagreement {np.median(np.abs(cx - gx)):.1f} px across, "
              f"{np.median(np.abs(cyv - gyv)):.1f} px down")
        sc["gen_track"] = gen

    print("\n" + "=" * 78)
    print("FRAMES CHOSEN  (turning points of the vertical path, then largest-travel fills)")
    print("=" * 78)
    for sc in scenes:
        cy = sc["track"][:, 1]
        bottom = sc["track"][:, 3]
        fps = sc["ctrl_info"].fps
        vh = sc["ctrl_info"].height
        idx, turns = choose_columns(cy, sc["valid"], N_COLS)
        sc.update(idx=idx, cy=cy, frame_h=vh, fps=fps)
        print(f"\n  {sc['title']}")
        print(f"    turning points found at frames {turns} "
              f"(t = {[round(i / fps, 3) for i in turns]} s)")
        print(f"    columns chosen: frames {idx} -> t = {[round(i / fps, 3) for i in idx]} s")
        print(f"    {'frame':>6s} {'t (s)':>7s} {'centre y':>9s} {'ball bottom':>12s} "
              f"{'height above floor':>19s} {'moved since prev':>17s} "
              f"{'ball vs bg (grey)':>18s}")
        prev = None
        for i in idx:
            moved = "-" if prev is None else f"{cy[i] - cy[prev]:+.1f} px"
            b, r = ball_contrast(sc["ctrl_frames"][i], sc["sil_frames"][i])
            print(f"    {i:6d} {i / fps:7.3f} {cy[i]:9.1f} {bottom[i]:12.1f} "
                  f"{vh - bottom[i]:19.1f} {moved:>17s} "
                  f"{abs(b - r):18.1f}")
            prev = i
        spread = cy[idx].max() - cy[idx].min()
        print(f"    the ball's centre spans {spread:.1f} px of the {vh} px frame height across "
              f"these {len(idx)} columns ({100 * spread / vh:.1f}% of the frame)")

    aspect = scenes[0]["ctrl_info"].width / scenes[0]["ctrl_info"].height

    print("\n" + "=" * 78)
    print("FIGURES  (one shared block template)")
    print("=" * 78)
    cases = [{"title": sc["title"], "ctrl": sc["ctrl_frames"], "gen": sc["gen_frames"],
              "idx": sc["idx"], "fps": sc["fps"], "cy": sc["cy"], "frame_h": sc["frame_h"]}
             for sc in scenes]
    p1 = build_figure(cases[:1], aspect, "fig_generation_pipeline")
    p2 = build_figure(cases, aspect, "fig_generation_cases")

    print("\n" + "=" * 78)
    print("OUTPUT FILES")
    print("=" * 78)
    from PIL import Image
    for p in p1 + p2:
        dims = ""
        if p.suffix == ".png":
            with Image.open(p) as im:
                dims = f"  {im.size[0]}x{im.size[1]} px"
        print(f"  {p}  {p.stat().st_size / 1024:8.1f} KB{dims}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
