#!/usr/bin/env python3
"""Rebuild the generation filmstrips (fig_generation_pipeline, fig_generation_cases) in the
report body font and at print resolution.

Same idea as make_fig_scene_breadth_bodyfont.py, for the two figures make_fig_generation.py
draws. It imports that maker untouched and overrides three things:

  1. Typeface  -- Source Sans -> Latin Modern Roman (the OpenType Computer Modern, i.e. the
     paper body face), so the case titles, row labels and timestamps are the same face as
     the body. Headings keep their emphasis as Latin Modern Roman Bold. Source Sans is not
     installed on this machine, so the stock maker cannot run here unchanged in any case.
  2. Resolution-- the stock save() embeds the control/generated frames at the default
     ~100-dpi device resolution; both figures are now written with frames embedded at
     600 dpi (PDF) and a 300-dpi review PNG, matching the scene and critic figures.
  3. Sources    -- the hard-coded /Users/wenzhuox absolute paths are repointed at the
     techreport_figure_data bundle. fig07_generation_cases/ holds all six clips both
     figures draw from (both scenes' depth control, silhouette, and generated video).

Everything else -- which frames are chosen (turning points of the tracked path), the
titles, the pairing self-checks -- is the bundle maker's, unchanged.

Run:  python3 figs/make_fig_generation_bodyfont.py
Idempotent: overwrites fig_generation_pipeline.{pdf,png} and fig_generation_cases.{pdf,png}.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPORT_FIGS = Path(__file__).resolve().parent
BUNDLE = Path("/Users/jigu/work/reports/techreport_figure_data")
MAKERS = BUNDLE / "maker_scripts"
# fig07 holds both scenes' clips (depth + silhouette control, generated); fig02 is a subset.
SRC = BUNDLE / "fig07_generation_cases"

LM_DIR = Path("/usr/local/texlive/2024/texmf-dist/fonts/opentype/public/lm")
LM_FACES = ["lmroman10-regular.otf", "lmroman10-bold.otf",
            "lmroman10-italic.otf", "lmroman10-bolditalic.otf"]
LM_FAMILY = "Latin Modern Roman"
PDF_FRAME_DPI = 600

sys.path.insert(0, str(MAKERS))
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
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    resolved = fm.findfont(fm.FontProperties(family=[LM_FAMILY]), fallback_to_default=False)
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
    for ext, kw in (("pdf", {"dpi": PDF_FRAME_DPI, "metadata": {"CreationDate": None}}),
                    ("png", {"dpi": 300})):
        p = REPORT_FIGS / f"{stem}.{ext}"
        fig.savefig(p, **kw)
        paths.append(p)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return paths


mfg.apply_style = apply_style_lm
mfg.semibold = semibold_lm
mfg.save = save_hi
mfg.OUT_DIR = REPORT_FIGS
mfg.CONTROLS = SRC / "control"
mfg.GENERATED = SRC / "generated"


if __name__ == "__main__":
    raise SystemExit(mfg.main())
