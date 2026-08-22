#!/usr/bin/env python3
"""Rebuild Figure 5 (release-point sweep) shorter, using the CORRECT row-layout maker.

The committed fig_injection_sweep.pdf is the three-panels-in-a-row layout produced by
techreport_figure_data/maker_scripts/make_fig_injection_sweep_row.py (Computer Modern,
6.5 x 2.95 in). The in-repo make_fig_injection_step.py draws a different, stacked layout,
so it must NOT be used here. This driver imports the row maker unchanged, lowers only its
figure height (BASE_H 2.95 -> 2.35) to remove the vertical whitespace, and copies the
output into the report's figs/. All data (HSV re-tracking of the sweep clips), the y-axis
transform, the panels, and the legend are the maker's, unchanged.

Run:  python3 figs/make_fig_injection_sweep_shorter.py
Idempotent: overwrites figs/fig_injection_sweep.{pdf,png}.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPORT_FIGS = Path(__file__).resolve().parent
MAKERS = Path("/Users/jigu/work/reports/techreport_figure_data/maker_scripts")

sys.path.insert(0, str(MAKERS))
import make_fig_injection_sweep_row as row  # noqa: E402

# lower the figure height only; proportions and margins scale with it
row.BASE_H = 2.35   # was 2.95

# draw() calls M.apply_style() first, which hard-errors when Source Sans is absent (it
# is, here) BEFORE draw() installs its own Computer Modern rcParams. Neutralize that
# check: draw() sets font.family=serif / cmr10 itself right after, matching the body.
row.M.apply_style = lambda: None

if __name__ == "__main__":
    outs = row.draw(scale=1.0)
    for p in outs:
        dst = REPORT_FIGS / p.name
        shutil.copy2(str(p), str(dst))
        print(f"[copy] {p} -> {dst}")
    raise SystemExit(0)
