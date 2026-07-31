#!/usr/bin/env python3
"""Build fig_critic_cases -- the six caught flaws, each with the frames a reader can check.

WHAT THIS FIGURE IS. Six clips from the evaluation core on which a deterministic
specialist -- not a yes/no vision-language judgement -- produced the accusation that the
matching protocol linked to a human annotator's sentence. One panel per clip.

WHICH FRAMES ARE SHOWN, AND WHY NOT HAND-PICKED. Every panel shows the FIRST and the LAST
decoded frame of the clip and nothing else. That rule is fixed for all six panels, is
stated in the caption, and cannot be tuned to flatter a case: the pair brackets the whole
clip, so a claim of the form "it rose / it never rolled / there are four of them" is
either visible across the bracket or it is not. Frames are decoded sequentially with no
seeking, so the last frame really is the last one.

The measurement text under each panel is copied from the run's own row (the tool's
provenance string, its required-versus-measured pair, and its evidence dict), never
paraphrased upward. Source of every number: critic_cases.json beside this script.

Run:  python3 /Users/wenzhuox/diffphy_psc/techreport/figs/make_fig_critic_cases.py

Idempotent: re-running overwrites fig_critic_cases.pdf and .png.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path("/Users/wenzhuox/diffphy_psc/techreport/figs")
CASES_JSON = OUT_DIR / "critic_cases.json"
VIDEO_DIR = Path("/Users/wenzhuox/diffphy_exp013/artifacts/runs/exp013/exp029/run_videos")

# ----------------------------------------------------------------------------- #
# the report's visual system (same palette and faces as make_fig_generation.py)
# ----------------------------------------------------------------------------- #
ADOBE_RED = "#FA0F00"
INK = "#1A1A1A"
GRAY = "#6E6E6E"
HAIRLINE = 0.6

FONT_DIRS = [
    "/Users/wenzhuox/Library/Caches/Tectonic/bundles/data/*",
    str(Path.home() / "Library/Fonts"),
    "/Library/Fonts",
    "/Library/Application Support/Adobe/*/*/Fonts",
]
FONT_PATTERNS = ("SourceSansPro-*.otf", "SourceSerifPro-*.otf", "SourceCodePro-*.otf")
SANS = ["Source Sans 3", "Source Sans Pro"]
SERIF = ["Source Serif 4", "Source Serif Pro"]
MONO = ["Source Code Pro"]


def apply_style() -> None:
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
    if not any(n in families for n in SANS):
        raise SystemExit("Source Sans not found and no fallback is acceptable here.")
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": SANS,
        "font.serif": SERIF, "font.monospace": MONO,
        "font.size": 7.0, "text.color": INK,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    for label, fams in (("sans", SANS), ("mono", MONO)):
        prop = fm.FontProperties(family=fams)
        path = fm.findfont(prop, fallback_to_default=False)
        print(f"[fonts] {label:<5s} -> {Path(path).name}")


def semibold(size: float):
    from matplotlib import font_manager as fm
    for root in FONT_DIRS:
        for base in glob.glob(root):
            hits = glob.glob(str(Path(base) / "SourceSansPro-Semibold.otf"))
            if hits:
                return fm.FontProperties(fname=sorted(hits)[0], size=size)
    return fm.FontProperties(family=SANS, size=size, weight="semibold")


def mono(size: float):
    from matplotlib import font_manager as fm
    return fm.FontProperties(family=MONO, size=size)


def read_all(path: Path) -> tuple[list[np.ndarray], float]:
    """Decode every frame sequentially (no seeking, so the last frame is the last frame)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if len(frames) < 2:
        raise SystemExit(f"decoded {len(frames)} frames from {path}")
    return frames, fps


# ----------------------------------------------------------------------------- #
# text layout. matplotlib's own wrap=True measures against the axes box, not the
# column we actually have, and silently overlapped all three blocks; so wrap here
# against renderer-measured widths and advance by a known line height.
# ----------------------------------------------------------------------------- #
class Measurer:
    def __init__(self):
        import matplotlib.pyplot as plt
        self._fig = plt.figure(figsize=(1, 1))
        self._fig.canvas.draw()
        self._r = self._fig.canvas.get_renderer()

    def width_in(self, s: str, fp) -> float:
        """Rendered width of `s` in inches."""
        w, _, _ = self._r.get_text_width_height_descent(s, fp, ismath=False)
        return w / self._fig.dpi

    def wrap(self, s: str, fp, avail_in: float) -> list[str]:
        out, line = [], ""
        for word in s.split():
            cand = f"{line} {word}".strip()
            if line and self.width_in(cand, fp) > avail_in:
                out.append(line)
                line = word
            else:
                line = cand
        if line:
            out.append(line)
        return out


def main() -> int:
    apply_style()
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    from matplotlib.patches import Rectangle

    cases = json.load(open(CASES_JSON))["cases"]
    print(f"[cases] {len(cases)} loaded from {CASES_JSON.name}")

    NROW, NCOL = 3, 2
    if len(cases) != NROW * NCOL:
        raise SystemExit(f"layout is {NROW}x{NCOL}; got {len(cases)} cases")

    FIG_W = 6.5                       # inches, the report text width (measured \linewidth
                                      # is 472.03pt = 6.56in; authoring at 6.5 like every
                                      # other figure keeps type at its intended size
                                      # instead of letting \includegraphics shrink it)
    PAD_L, GUT = 0.055, 0.055         # left inset inside a cell, gutter between frames
    CELL_W = FIG_W / NCOL
    FRAME_W = (CELL_W - 2 * PAD_L - GUT) / 2
    FRAME_H = FRAME_W * 9 / 16
    HEAD_H = 0.135                    # heading band above the frames
    LAB_DY, TXT_DY = 0.088, 0.185     # frame caption baseline, then the text block top
    TXT_W = 2 * FRAME_W + GUT         # the text column spans both frames
    KEY_W = 0.40                      # "annotator / measured / verdict" gutter
    BODY_PT = 5.9
    LH = 1.30 * BODY_PT / 72.0        # line advance, inches
    BLOCK_GAP = 0.040
    BOT_PAD = 0.075

    body = fm.FontProperties(family=SANS, size=BODY_PT)
    body_mono = mono(BODY_PT)
    m = Measurer()

    # wrap every block first, so one FOOT_H fits the tallest cell and no cell overlaps
    blocks = []
    for case in cases:
        rows = [("annotator", m.wrap(case["human"], body, TXT_W - KEY_W), body, INK),
                ("measured", m.wrap(case["measured"], body_mono, TXT_W - KEY_W),
                 body_mono, INK),
                ("verdict", m.wrap(case["verdict_line"], body, TXT_W - KEY_W), body,
                 ADOBE_RED)]
        blocks.append(rows)
    max_lines = max(sum(len(r[1]) for r in b) for b in blocks)
    FOOT_H = TXT_DY + max_lines * LH + 2 * BLOCK_GAP + BOT_PAD
    CELL_H = HEAD_H + FRAME_H + FOOT_H
    FIG_H = NROW * CELL_H
    print(f"[layout] cell {CELL_W:.2f}x{CELL_H:.2f} in, frame {FRAME_W:.2f}x{FRAME_H:.2f}, "
          f"tallest text {max_lines} lines -> figure {FIG_W:.2f}x{FIG_H:.2f} in")

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    X = lambda v: v / FIG_W  # noqa: E731
    Y = lambda v: v / FIG_H  # noqa: E731

    for i, (case, rows) in enumerate(zip(cases, blocks)):
        row, col = divmod(i, NCOL)
        vid = VIDEO_DIR / f"{case['clip8']}.mp4"
        frames, fps = read_all(vid)
        first, last = frames[0], frames[-1]
        t_last = (len(frames) - 1) / fps if fps else float("nan")
        print(f"[{case['clip8']}] {len(frames)} frames @ {fps:.3g} fps -> "
              f"first t=0.00s, last t={t_last:.2f}s  ({vid.name})")

        x0 = col * CELL_W + PAD_L
        y_top = FIG_H - row * CELL_H

        # heading: clip id, then which instrument answered (placed past the measured id width)
        cid_fp = mono(7.2)
        fig.text(X(x0), Y(y_top - 0.095), case["clip8"], fontproperties=cid_fp,
                 color=INK, va="baseline")
        fig.text(X(x0 + m.width_in(case["clip8"], cid_fp) + 0.075), Y(y_top - 0.095),
                 case["instrument"], fontproperties=semibold(7.2), color=ADOBE_RED,
                 va="baseline")

        y_fr = y_top - HEAD_H - FRAME_H
        for j, (fr, lab) in enumerate(((first, "first frame  0.00 s"),
                                       (last, f"last frame  {t_last:.2f} s"))):
            fx = x0 + j * (FRAME_W + GUT)
            ax = fig.add_axes([X(fx), Y(y_fr), X(FRAME_W), Y(FRAME_H)])
            ax.imshow(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(GRAY); s.set_linewidth(HAIRLINE)
            fig.text(X(fx), Y(y_fr - LAB_DY), lab, fontproperties=mono(5.6),
                     color=GRAY, va="baseline")

            # A text case is only checkable if the reader can READ the render. Where a case
            # declares one, magnify that region of the SAME last frame in place and box the
            # source, so the crop adds resolution and not a different picture.
            z = case.get("zoom")
            if z and j == 1:
                zx1, zy1, zx2, zy2 = (int(v) for v in z)
                zw = FRAME_W * 0.94
                zh = zw * (zy2 - zy1) / (zx2 - zx1)
                za = fig.add_axes([X(fx + (FRAME_W - zw) / 2), Y(y_fr + 0.028),
                                   X(zw), Y(zh)], zorder=6)
                za.imshow(cv2.cvtColor(fr[zy1:zy2, zx1:zx2], cv2.COLOR_BGR2RGB))
                za.set_xticks([]); za.set_yticks([])
                for s in za.spines.values():
                    s.set_color(ADOBE_RED); s.set_linewidth(0.7)
                ax.add_patch(Rectangle((zx1, zy1), zx2 - zx1, zy2 - zy1, fill=False,
                                       edgecolor=ADOBE_RED, linewidth=0.7))

        # the human's sentence, then the tool's own numbers, then the verdict
        y = y_top - HEAD_H - FRAME_H - TXT_DY
        for key, lines, fp, colour in rows:
            fig.text(X(x0), Y(y), key, fontproperties=semibold(BODY_PT), color=GRAY,
                     va="top")
            for k, line in enumerate(lines):
                fig.text(X(x0 + KEY_W), Y(y - k * LH), line, fontproperties=fp,
                         color=colour, va="top")
            y -= len(lines) * LH + BLOCK_GAP

    for out in (OUT_DIR / "fig_critic_cases.pdf", OUT_DIR / "fig_critic_cases.png"):
        fig.savefig(out, dpi=300 if out.suffix == ".png" else None,
                    bbox_inches=None, pad_inches=0)
        print(f"[write] {out}  ({out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
