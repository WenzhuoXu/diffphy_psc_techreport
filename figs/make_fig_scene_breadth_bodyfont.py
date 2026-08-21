#!/usr/bin/env python3
"""Rebuild Figures 8 and 9 (scene breadth) so their type matches the report body.

The report body is set in Computer Modern Roman (the class overrides only \\sfdefault
and \\ttdefault, never \\rmdefault). The stock scene-breadth maker draws every label in
Source Sans, which (a) does not match the body and (b) is not even installed on this
machine, so the maker cannot run here unchanged. This driver imports the bundle makers
untouched and overrides three things:

  1. Typeface  -- Source Sans -> Latin Modern Roman, the OpenType form of Computer
     Modern, so figure text is the same face as the surrounding paragraphs. Headings
     keep their emphasis as Latin Modern Roman *Bold* rather than sans semibold.
  2. Heading   -- the "Beyond a single ball" title on the contact sheet is dropped
     (the LaTeX \\caption already says what the figure is); the "N of 66 scene kinds"
     note moves to the left edge so the header still reads as one line above the rule.
  3. Resolution-- the stock save() embeds the video frames into the PDF at the default
     ~100-dpi device resolution, which is why the tiles look soft. Both figures are now
     written with the frames embedded at 600 dpi (PDF) and a 300-dpi PNG for review.

Everything else -- which frame each tile shows, the caption text, the layout geometry,
every measurement and self-check -- is the bundle maker's, unchanged. Assets are read
from techreport_figure_data/fig08_scene_breadth/.

Run:  python3 figs/make_fig_scene_breadth_bodyfont.py
Idempotent: overwrites fig_scene_breadth.{pdf,png} and fig_scene_control_pairs.{pdf,png}.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# --------------------------------------------------------------------------- paths
REPORT_FIGS = Path(__file__).resolve().parent
BUNDLE = Path("/Users/jigu/work/reports/techreport_figure_data")
SCENE = BUNDLE / "fig08_scene_breadth"
MAKERS = BUNDLE / "maker_scripts"

# The OpenType Computer Modern shipped with MacTeX; this is the body face.
LM_DIR = Path("/usr/local/texlive/2024/texmf-dist/fonts/opentype/public/lm")
LM_FACES = ["lmroman10-regular.otf", "lmroman10-bold.otf",
            "lmroman10-italic.otf", "lmroman10-bolditalic.otf"]
LM_FAMILY = "Latin Modern Roman"

sys.path.insert(0, str(MAKERS))

# --------------------------------------------------------------------------- overrides
import make_fig_generation as mfg  # noqa: E402


def apply_style_lm() -> None:
    """Register Latin Modern Roman and declare it as the figure's serif body face."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm

    for face in LM_FACES:
        p = LM_DIR / face
        if not p.exists():
            raise SystemExit(f"missing body font {p} -- cannot match the paper's Computer Modern")
        fm.fontManager.addfont(str(p))

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [LM_FAMILY],
        "font.size": 8.0,
        "text.color": mfg.INK,
        "axes.edgecolor": mfg.GRAY,
        "axes.linewidth": mfg.HAIRLINE,
        "axes.grid": False,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "mathtext.fontset": "cm",     # any incidental math also in Computer Modern
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    prop = fm.FontProperties(family=[LM_FAMILY])
    resolved = fm.findfont(prop, fallback_to_default=False)
    print(f"[fonts] body face -> {Path(resolved).name} "
          f"({fm.FontProperties(fname=resolved).get_name()})")


def semibold_lm(size: float):
    """Heading weight, in the body family: Latin Modern Roman Bold."""
    from matplotlib import font_manager as fm
    return fm.FontProperties(fname=str(LM_DIR / "lmroman10-bold.otf"), size=size)


def save_hi(fig, stem: str):
    """Write the PDF with frames embedded at 600 dpi, plus a 300-dpi PNG for review."""
    REPORT_FIGS.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext, kw in (("pdf", {"dpi": 600, "metadata": {"CreationDate": None}}),
                    ("png", {"dpi": 300})):
        p = REPORT_FIGS / f"{stem}.{ext}"
        fig.savefig(p, **kw)
        paths.append(p)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return paths


# Drop the contact-sheet title; left-align the "N of 66 scene kinds" note in its place.
_orig_layout_text = mfg.Layout.text


def _layout_text(self, x_in, y_in, s, **kw):
    if s == "Beyond a single ball":
        return _orig_layout_text(self, x_in, y_in, "", **kw)      # heading removed
    if isinstance(s, str) and s.endswith("scene kinds"):
        kw = {**kw, "ha": "left"}
        return _orig_layout_text(self, 0.0, y_in, s, **kw)        # move to left edge
    return _orig_layout_text(self, x_in, y_in, s, **kw)


mfg.apply_style = apply_style_lm
mfg.semibold = semibold_lm
mfg.save = save_hi
mfg.Layout.text = _layout_text
mfg.OUT_DIR = REPORT_FIGS

# Tighten Figure 9's vertical whitespace. These module constants are read at call
# time by build_figure/block_h/draw_block, so overriding them here shrinks the
# per-case heading band, the control->generated row gap, the frames->timestamp gap,
# the timestamp band, and the gap between the three cases. Figure 8 is unaffected: it
# draws from the separate A_* constants in make_fig_scene_breadth. HEAD_H stays above
# HEAD_RULE (0.160) so the frames never climb over the heading's hairline rule.
mfg.HEAD_H = 0.200      # was 0.235  (heading band; 0.040 in clearance over the rule)
mfg.ROW_GAP = 0.045     # was 0.060  (control row -> generated row)
mfg.TS_GAP = 0.065      # was 0.085  (frames -> timestamps)
mfg.TS_H = 0.095        # was 0.105  (timestamp band)
mfg.BLOCK_GAP = 0.105   # was 0.170  (between stacked cases)

# import AFTER patching so scene_breadth's `from make_fig_generation import ...`
# binds to the overridden apply_style / semibold / save.
import make_fig_scene_breadth as sb  # noqa: E402

# repoint every hard-coded source path at the bundle
sb.OUT_DIR = REPORT_FIGS
sb.GEN = SCENE / "generated"
sb.CTRL = SCENE / "control"
sb.EYECHECK = SCENE / "eyecheck.jsonl"
sb.METRICS = SCENE / "metrics_all.jsonl"
sb.BUILDERS = SCENE / "gen_scene_library.py"


if __name__ == "__main__":
    raise SystemExit(sb.main())
