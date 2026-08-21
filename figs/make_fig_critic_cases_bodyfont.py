#!/usr/bin/env python3
"""Rebuild Figure 10 (the six tool-caught flaw cases) hi-res, body-font, one sample per row.

This forks the bundle maker's main() to relayout the figure and restyle its text, while
reusing the maker's own frame reader and text measurer. The measurement text is copied
verbatim from critic_cases.json; only its presentation changes.

Layout -- one sample per full-width row, six rows stacked. Each row is a five-frame
filmstrip sampled uniformly across the whole clip (frames 0, 1/4, 1/2, 3/4, last), so a
reader sees the motion develop rather than only its endpoints; the old panel showed just
the first and last frame side by side. Because each strip now spans the page, the
annotator / measured / verdict lines sit flat and full-width beneath it, one line each,
instead of wrapping in a narrow half-page column.

Text styling, unchanged from the prior revision:
  * Latin Modern Roman throughout (the OpenType Computer Modern = the paper body face);
    the measured tool output is proportional Roman, not monospace, so its letter spacing
    matches the body. Source Sans is not installed here in any case.
  * First letters capitalized: the keys (Annotator / Measured / Verdict), the frame time
    labels, the instrument names, and each block's first word.
  * No raw clip hash above the strip; the instrument name that caught the flaw heads the
    row. A light CVD-checked colour tells the keys apart: blue Annotator, teal Measured,
    red Verdict (and the instrument name).

Resolution: frames embedded in the PDF at 600 dpi, matching Figures 7 and 8; review PNG
at 300 dpi.

Run:  python3 figs/make_fig_critic_cases_bodyfont.py
Idempotent: overwrites fig_critic_cases.{pdf,png}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

REPORT_FIGS = Path(__file__).resolve().parent
BUNDLE = Path("/Users/jigu/work/reports/techreport_figure_data")
CASE = BUNDLE / "fig16_critic_cases"
MAKERS = BUNDLE / "maker_scripts"

LM_DIR = Path("/usr/local/texlive/2024/texmf-dist/fonts/opentype/public/lm")
LM_ROMAN = ["lmroman10-regular.otf", "lmroman10-bold.otf",
            "lmroman10-italic.otf", "lmroman10-bolditalic.otf"]
ROMAN_FAMILY = "Latin Modern Roman"
PDF_FRAME_DPI = 600

# Row-key colours (validated CVD-safe with the dataviz palette validator, light mode:
# worst adjacent ΔE 14.8 deutan, normal-vision floor ΔE 21.0, all >= 3:1 contrast).
BLUE = "#2563C9"   # Annotator -- the human's sentence
TEAL = "#2A9D8F"   # Measured  -- the tool's own numbers

sys.path.insert(0, str(MAKERS))
import make_fig_critic_cases as cc  # noqa: E402

# repoint the hard-coded source paths at the bundle
cc.OUT_DIR = REPORT_FIGS
cc.CASES_JSON = CASE / "critic_cases.json"
cc.VIDEO_DIR = CASE / "clips"


def register_roman():
    from matplotlib import font_manager as fm
    for face in LM_ROMAN:
        p = LM_DIR / face
        if not p.exists():
            raise SystemExit(f"missing font {p} -- cannot match the paper's Computer Modern")
        fm.fontManager.addfont(str(p))


def roman(size: float, bold: bool = False):
    from matplotlib import font_manager as fm
    face = "lmroman10-bold.otf" if bold else "lmroman10-regular.otf"
    return fm.FontProperties(fname=str(LM_DIR / face), size=size)


def cap(s: str) -> str:
    """Capitalize the first character only, leaving the rest (e.g. 'OCR', 'vs.') intact."""
    s = s.strip()
    return s[:1].upper() + s[1:] if s else s


def sample_indices(n: int, k: int = 5) -> list[int]:
    """k frame indices spread uniformly across a clip of n frames (first .. last inclusive)."""
    if n <= k:
        return list(range(n))
    return [round(t * (n - 1) / (k - 1)) for t in range(k)]


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    from matplotlib.patches import Rectangle

    register_roman()
    plt.rcParams.update({
        "font.family": "serif", "font.serif": [ROMAN_FAMILY],
        "font.size": 7.0, "text.color": cc.INK,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "mathtext.fontset": "cm", "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    print(f"[fonts] body face -> {Path(fm.findfont(fm.FontProperties(family=[ROMAN_FAMILY]), fallback_to_default=False)).name}")

    cases = json.load(open(cc.CASES_JSON))["cases"]
    print(f"[cases] {len(cases)} loaded from {cc.CASES_JSON.name}")

    # ---- layout: one sample per full-width row, six rows stacked ----
    FIG_W = 6.5
    NFR = 5                              # frames per filmstrip
    PAD_L = 0.02                         # inset from the figure edge
    GUT = 0.05                           # gap between frames in a strip
    STRIP_W = FIG_W - 2 * PAD_L
    FRAME_W = (STRIP_W - (NFR - 1) * GUT) / NFR
    FRAME_H = FRAME_W * 9 / 16
    HEAD_DY = 0.095                      # instrument heading baseline above the strip
    HEAD_H = 0.135                       # heading band
    TS_DY = 0.088                        # timestamp baseline below the strip
    TS_H = 0.115                         # timestamp band
    TXT_TOP = 0.055                      # gap timestamp band -> first text line
    KEY_W = 0.58                         # bold key column ("Annotator") -> its text
    BODY_PT = 5.8                        # 5.8, not 5.9: keeps the longest measured line to a
                                         # single full-width line (5.65 of 5.90 in available)
    LH = 1.30 * BODY_PT / 72.0           # text line advance, inches
    ROW_PAD = 0.095                      # gap between one sample row and the next
    print(f"[layout] {len(cases)} full-width rows, {NFR} frames each; frame "
          f"{FRAME_W:.3f}x{FRAME_H:.3f} in (strip {STRIP_W:.2f} in)")

    body = roman(BODY_PT)
    key_fp = roman(BODY_PT, bold=True)
    m = cc.Measurer()

    # A line that is a hair too long wraps to two, so reserve the row's text band for the
    # worst case measured across all six samples rather than assuming three flat lines.
    def block_lines(case):
        return sum(len(m.wrap(cap(case[k]), body, FIG_W - PAD_L - KEY_W))
                   for k in ("human", "measured", "verdict_line"))
    max_lines = max(block_lines(c) for c in cases)
    TEXT_H = max_lines * LH
    ROW_H = HEAD_H + FRAME_H + TS_H + TXT_TOP + TEXT_H + ROW_PAD
    FIG_H = len(cases) * ROW_H
    print(f"[layout] worst text block {max_lines} lines -> row {ROW_H:.3f} in, "
          f"figure {FIG_W:.2f}x{FIG_H:.2f} in"
          + ("  [warn] exceeds 9.13 in text height" if FIG_H > 9.13 else ""))

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    X = lambda v: v / FIG_W  # noqa: E731
    Y = lambda v: v / FIG_H  # noqa: E731

    for i, case in enumerate(cases):
        vid = cc.VIDEO_DIR / f"{case['clip8']}.mp4"
        frames, fps = cc.read_all(vid)
        idx = sample_indices(len(frames), NFR)
        times = [k / fps if fps else float("nan") for k in idx]
        print(f"[{case['clip8']}] {len(frames)} frames @ {fps:.3g} fps -> "
              f"sampled {idx} (t = {[round(t, 2) for t in times]} s)")

        y_top = FIG_H - i * ROW_H

        # heading: what the clip shows, then the specialist that caught the flaw (no clip
        # id). The subject is ink so the row names its own footage; the instrument is the
        # accent colour, and carrying the subject also tells the two "counter" rows apart.
        head_fp = roman(7.6, bold=True)
        subj = cap(case["subject"]) + "  —  "   # subject, em dash, instrument
        fig.text(X(PAD_L), Y(y_top - HEAD_DY), subj,
                 fontproperties=head_fp, color=cc.INK, va="baseline")
        subj_w = m.width_in(subj, head_fp)
        fig.text(X(PAD_L + subj_w), Y(y_top - HEAD_DY), cap(case["instrument"]),
                 fontproperties=head_fp, color=cc.ADOBE_RED, va="baseline")

        # the five-frame filmstrip
        y_fr = y_top - HEAD_H - FRAME_H
        z = case.get("zoom")
        for c, k in enumerate(idx):
            fx = PAD_L + c * (FRAME_W + GUT)
            ax = fig.add_axes([X(fx), Y(y_fr), X(FRAME_W), Y(FRAME_H)])
            ax.imshow(cv2.cvtColor(frames[k], cv2.COLOR_BGR2RGB))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(cc.GRAY); s.set_linewidth(cc.HAIRLINE)
            fig.text(X(fx + FRAME_W / 2), Y(y_fr - TS_DY), f"{times[c]:.2f} s",
                     fontproperties=roman(5.6), color=cc.GRAY, va="baseline", ha="center")

            # text case: box the region on the last frame and magnify it just above the strip
            if z and c == NFR - 1:
                zx1, zy1, zx2, zy2 = (int(v) for v in z)
                zw = FRAME_W * 1.4
                zh = zw * (zy2 - zy1) / (zx2 - zx1)
                za = fig.add_axes([X(fx + FRAME_W - zw), Y(y_fr + FRAME_H - zh + 0.002),
                                   X(zw), Y(zh)], zorder=6)
                za.imshow(cv2.cvtColor(frames[k][zy1:zy2, zx1:zx2], cv2.COLOR_BGR2RGB))
                za.set_xticks([]); za.set_yticks([])
                for s in za.spines.values():
                    s.set_color(cc.ADOBE_RED); s.set_linewidth(0.7)
                ax.add_patch(Rectangle((zx1, zy1), zx2 - zx1, zy2 - zy1, fill=False,
                                       edgecolor=cc.ADOBE_RED, linewidth=0.7))

        # flat, full-width text: one line each, key coloured, spanning the page
        rows = [("Annotator", cap(case["human"]), BLUE, cc.INK),
                ("Measured", cap(case["measured"]), TEAL, cc.INK),
                ("Verdict", cap(case["verdict_line"]), cc.ADOBE_RED, cc.ADOBE_RED)]
        y = y_fr - TS_H - TXT_TOP
        for key, text, key_colour, body_colour in rows:
            fig.text(X(PAD_L), Y(y), key, fontproperties=key_fp, color=key_colour, va="top")
            lines = m.wrap(text, body, FIG_W - PAD_L - KEY_W)
            for j, line in enumerate(lines):
                fig.text(X(PAD_L + KEY_W), Y(y - j * LH), line, fontproperties=body,
                         color=body_colour, va="top")
            y -= max(1, len(lines)) * LH

    for out in (REPORT_FIGS / "fig_critic_cases.pdf", REPORT_FIGS / "fig_critic_cases.png"):
        fig.savefig(out, dpi=PDF_FRAME_DPI if out.suffix == ".pdf" else 300,
                    bbox_inches=None, pad_inches=0)
        print(f"[write] {out}  ({out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
