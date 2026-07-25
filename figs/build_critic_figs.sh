#!/usr/bin/env bash
# =====================================================================
# build_critic_figs.sh -- compile and preview the three critic schematics.
#
# Idempotent and re-runnable. For each of
#     fig_critic_pipeline.tex
#     fig_plan_example.tex
#     fig_graph_construction.tex
# it (1) compiles the figure inside a tiny standalone wrapper that mimics
# main.tex's font and colour setup, (2) writes a tightly cropped, vector
# <name>.pdf next to the .tex, and (3) renders <name>.png at 200 dpi for
# eyeball review. It then compiles ALL figures inside a document that loads
# the report's real adobe-techreport.sty -- each figure twice -- to prove the
# TikZ library and style definitions do not clash and are safe to \input more
# than once, and that no box overflows.
#
# The .tex files are what main.tex \inputs; the .pdf/.png are review artifacts.
#
#   bash /Users/wenzhuox/diffphy_psc/techreport/figs/build_critic_figs.sh
#
# Requires: tectonic, pdftoppm (poppler), python3 with PyMuPDF (fitz).
# =====================================================================
set -euo pipefail

FIGS="/Users/wenzhuox/diffphy_psc/techreport/figs"
REPORT="/Users/wenzhuox/diffphy_psc/techreport"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/critic_figs.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

TECTONIC="${TECTONIC:-/opt/homebrew/bin/tectonic}"
PDFTOPPM="${PDFTOPPM:-/opt/homebrew/bin/pdftoppm}"

cp "$FIGS/fig_critic_pipeline.tex" "$FIGS/fig_plan_example.tex" \
   "$FIGS/fig_graph_construction.tex" "$WORK/"
cp "$REPORT/adobe-techreport.sty" "$WORK/"

# --- 0. reprint every quoted value, straight from the benchmark file ---
# fig_critic_pipeline.tex and fig_graph_construction.tex contain no quantities
# at all. fig_plan_example.tex quotes one clip's real human annotation; this
# prints it so the figure can be checked against the source, and fails loudly
# if the record ever changes.
python3 - <<'PYEOF'
import json
import sys

GOLD = ("/Users/wenzhuox/diffphy_exp013/artifacts/runs/exp013/"
        "gold_v1/gold_core_v1.json")
ITEM = "3435a629-af23-50b3-bc74-8b34085e958d"

gold = json.loads(open(GOLD).read())
clips = gold["clips"]
clip = next(c for c in clips if c["item_id"] == ITEM)
flaws = clip["flaws"]

print("provenance of every value quoted in fig_plan_example.tex")
print("  source: %s" % GOLD)
print("  clips in the physics-focused evaluation set : %d" % len(clips))
print("  worked-example clip, flaws recorded on it   : %d" % len(flaws))
for f in flaws:
    print("  human-marked prompt fragment : %r" % f["span"])
    print("  human rationale              : %r" % f["reasoning"])
    print("  human category / severity    : %s / %s of 5"
          % (f["category"], f["severity"]))

# the figure asserts exactly these; fail loudly if the source record moves
f = flaws[0]
checks = [
    ("claim sentence", f["span"],
     "One ball is tossed in, triggering a cascade of 1,000 traps."),
    ("count seen in the video", "63" in f["reasoning"], True),
    ("flaw category", f["category"], "count"),
    ("severity", int(f["severity"]), 3),
    ("only flaw on the clip", len(flaws), 1),
]
bad = [n for n, got, want in checks if got != want]
if bad:
    print("FAIL: the figure no longer matches the source for: %s" % ", ".join(bad))
    sys.exit(1)
print("  all values in the figure match the source record")
print()
PYEOF

# --- 1. each figure on its own, cropped to its own bounding box -------
for fig in fig_critic_pipeline fig_plan_example fig_graph_construction; do
  cat > "$WORK/standalone_$fig.tex" <<EOF
% Standalone wrapper: same fonts and brand colours as the report.
\\documentclass[11pt,letterpaper]{article}
\\usepackage[T1]{fontenc}
\\usepackage{sourceserifpro}
\\usepackage[semibold]{sourcesanspro}
\\usepackage{sourcecodepro}
\\usepackage{xcolor}
\\definecolor{adobered}{HTML}{FA0F00}
\\definecolor{adobeink}{HTML}{1A1A1A}
\\definecolor{adobegray}{HTML}{6E6E6E}
\\usepackage{tikz}
\\usetikzlibrary{arrows.meta,positioning,calc}
\\usepackage[margin=3mm,paperwidth=180mm,paperheight=260mm]{geometry}
\\pagestyle{empty}
\\begin{document}
\\noindent\\input{$fig.tex}
\\end{document}
EOF
  ( cd "$WORK" && "$TECTONIC" -X compile "standalone_$fig.tex" --outdir "$WORK" >/dev/null 2>&1 )

  python3 - "$WORK/standalone_$fig.pdf" "$FIGS/$fig.pdf" <<'PYEOF'
"""Re-issue the drawing on a page shrink-wrapped to its own ink.

Both the media box and the crop box end up equal to the figure's bounding
box, so every consumer (LaTeX, poppler, a PDF viewer) sees the same tight
page. The content stays vector -- nothing is rasterized.
"""
import sys
import fitz

src, dst = sys.argv[1], sys.argv[2]
doc = fitz.open(src)
page = doc[0]

# union of every drawing path and every text block on the page
r = fitz.Rect(page.rect.x1, page.rect.y1, page.rect.x0, page.rect.y0)
for d in page.get_drawings():
    r |= d["rect"]
for b in page.get_text("blocks"):
    r |= fitz.Rect(b[:4])
pad = 2.0
r = fitz.Rect(r.x0 - pad, r.y0 - pad, r.x1 + pad, r.y1 + pad) & page.rect

out = fitz.open()
new = out.new_page(width=r.width, height=r.height)
new.show_pdf_page(new.rect, doc, 0, clip=r)
out.save(dst, garbage=4, deflate=True)
print("  %s  %.2f x %.2f in" % (dst, r.width / 72, r.height / 72))
PYEOF

  "$PDFTOPPM" -r 200 -png -singlefile -cropbox "$FIGS/$fig.pdf" "$FIGS/$fig"
  echo "wrote $FIGS/$fig.pdf and $FIGS/$fig.png"
done

# --- 2. every figure inside the report's real style, twice each -------
cat > "$WORK/with_style.tex" <<'EOF'
% Mimics main.tex: loads adobe-techreport.sty (tcolorbox/pgf, listings,
% hyperref, cleveref) and \inputs EVERY figure TWICE, to prove no style or
% TikZ-library clash and no double-definition error.
\documentclass[11pt,letterpaper]{article}
\usepackage{adobe-techreport}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,calc}
\begin{document}
\begin{figure}[t]\centering\input{fig_critic_pipeline.tex}
  \caption{Critic dataflow.}\label{fig:critic}\end{figure}
\begin{figure}[t]\centering\input{fig_plan_example.tex}
  \caption{A worked example.}\label{fig:plan}\end{figure}
\clearpage
\begin{figure}[t]\centering\input{fig_graph_construction.tex}
  \caption{Plan construction.}\label{fig:gc}\end{figure}
\clearpage
\begin{figure}[t]\centering\input{fig_plan_example.tex}
  \caption{Repeat.}\label{fig:plan2}\end{figure}
\begin{figure}[t]\centering\input{fig_critic_pipeline.tex}
  \caption{Repeat.}\label{fig:critic2}\end{figure}
\clearpage
\begin{figure}[t]\centering\input{fig_graph_construction.tex}
  \caption{Repeat.}\label{fig:gc2}\end{figure}
\Cref{fig:critic}, \cref{fig:plan}, \cref{fig:gc}.
\end{document}
EOF
( cd "$WORK" && "$TECTONIC" -X compile with_style.tex --outdir "$WORK" --keep-logs >/dev/null 2>&1 )
BAD=$(grep -c 'Overfull\|Underfull' "$WORK/with_style.log" || true)
echo "all figures, twice each, under adobe-techreport.sty: ${BAD} overfull/underfull boxes"
[ "$BAD" -eq 0 ] || { echo "FAIL: over/underfull boxes present"; exit 1; }

# --- 3. each figure must fit the report's text width, and must come out
#        the SAME width every time it is included (a TikZ library reloaded
#        inside a figure group silently accumulates and widens the picture) ---
cat > "$WORK/measure.tex" <<'EOF'
\documentclass[11pt,letterpaper]{article}
\usepackage{adobe-techreport}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,calc}
\newsavebox{\bx}
\begin{document}
\typeout{MEASURE textwidth \the\textwidth}
\savebox{\bx}{\input{fig_critic_pipeline.tex}}\typeout{MEASURE critic1 \the\wd\bx}
\savebox{\bx}{\input{fig_plan_example.tex}}\typeout{MEASURE plan1 \the\wd\bx}
\savebox{\bx}{\input{fig_graph_construction.tex}}\typeout{MEASURE build1 \the\wd\bx}
\savebox{\bx}{\input{fig_plan_example.tex}}\typeout{MEASURE plan2 \the\wd\bx}
\savebox{\bx}{\input{fig_critic_pipeline.tex}}\typeout{MEASURE critic2 \the\wd\bx}
\savebox{\bx}{\input{fig_graph_construction.tex}}\typeout{MEASURE build2 \the\wd\bx}
x
\end{document}
EOF
( cd "$WORK" && "$TECTONIC" -X compile measure.tex --outdir "$WORK" --keep-logs >/dev/null 2>&1 )
python3 - "$WORK/measure.log" <<'PYEOF'
import re
import sys

vals = dict(re.findall(r"MEASURE (\w+) ([\d.]+)pt", open(sys.argv[1]).read()))
tw = float(vals["textwidth"])
ok = True
for name in ("critic", "plan", "build"):
    w1, w2 = float(vals[name + "1"]), float(vals[name + "2"])
    fits = w1 <= tw
    same = abs(w1 - w2) < 0.01
    print("  %-6s width %.1fpt of %.1fpt available -> %s; identical on "
          "re-inclusion -> %s" % (name, w1, tw, "fits" if fits else "TOO WIDE",
                                  "yes" if same else "NO (%.1fpt)" % w2))
    ok = ok and fits and same
sys.exit(0 if ok else 1)
PYEOF
echo "OK"
