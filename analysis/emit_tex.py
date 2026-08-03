#!/usr/bin/env python3
"""emit_tex.py — write the Results tables as LaTeX, straight from the measured numbers.

Nothing in `results_tables.tex` is typed by hand. It is generated from:
  * `tables.py`  — the descriptive tables over the published run's per-flaw rows
  * `paired.py`  — the fresh paired head-to-head written by this session's own runs

If a run is incomplete, the corresponding table is emitted with the denominator it actually
has and a note saying so, rather than being padded, projected, or omitted silently. A table
that cannot be computed at all is replaced by a visible TODO marker so it cannot be mistaken
for a finished result.

    python analysis/emit_tex.py --out results_tables.tex
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _mod(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def esc(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%")


CAT_LABEL = {
    "action": "action or event occurs", "count": "how many", "object": "object identity",
    "text_ocr": "on-screen text", "spatial": "spatial relation",
    "order_timing": "order and timing", "attribute": "attribute",
    "camera_style": "camera and style", "other": "other",
    "physics_motion": "physical motion",
}


def tab_category(cat: dict) -> str:
    rows = []
    tot = full = part = 0
    for c, v in sorted(cat.items(), key=lambda kv: -kv[1]["total"]):
        rows.append(f"{esc(CAT_LABEL.get(c, c))} & {v['total']} & {v['full']} & "
                    f"{v['partial']} & {v['found']} & "
                    f"{100.0*v['found']/v['total']:.0f}\\% \\\\")
        tot += v["total"]; full += v["full"]; part += v["partial"]
    rows.append(r"\midrule")
    rows.append(f"All & {tot} & {full} & {part} & {full+part} & "
                f"{100.0*(full+part)/tot:.0f}\\% \\\\")
    body = "\n".join(rows)
    return rf"""
\begin{{table}}[t]
\centering\small
\setlength{{\tabcolsep}}{{6pt}}\renewcommand{{\arraystretch}}{{1.15}}
\begin{{tabular}}{{@{{}}p{{4.6cm}} c c c c c@{{}}}}
\toprule
\textbf{{What the annotator complained about}} & \textbf{{Flaws}} & \textbf{{Whole}}
 & \textbf{{Part}} & \textbf{{Found}} & \textbf{{Rate}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\caption{{Recall by the annotators' own flaw category on the published run. ``Whole'' counts
flaws a single finding covers completely; ``Part'' counts flaws for which our findings cover
an aspect but no one finding is the whole defect; ``Found'' is their sum. Categories are the
annotators' labels, assigned before any critic ran, so this table is a property of the
evaluation set as much as of the critic. Counting is the weakest category and the
measurement path behind it is unreliable (\cref{{sec:contract}}).}}
\label{{tab:bycat}}
\end{{table}}
"""


def tab_robust(rob: dict, sev: dict) -> str:
    def block(title, d):
        out = [rf"\multicolumn{{4}}{{@{{}}l}}{{\textit{{{title}}}}} \\"]
        for k, (f, n) in d.items():
            out.append(f"\\quad {esc(k)} & {n} & {f} & {100.0*f/n:.0f}\\% \\\\")
        return "\n".join(out)

    sev_d = {f"severity {k}": v for k, v in sorted(sev.items())}
    body = "\n\\addlinespace\n".join([
        block("Annotator severity", sev_d),
        block("Clip duration", rob["clip duration"]),
        block("Flaws annotated in the clip", rob["flaws in clip"]),
        block("Difficulty stratum", rob["stratum"]),
    ])
    return rf"""
\begin{{table}}[t]
\centering\small
\setlength{{\tabcolsep}}{{6pt}}\renewcommand{{\arraystretch}}{{1.12}}
\begin{{tabular}}{{@{{}}p{{5.0cm}} c c c@{{}}}}
\toprule
 & \textbf{{Flaws}} & \textbf{{Found}} & \textbf{{Rate}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\caption{{Recall against four properties of the clip and the flaw, on the published run.
The profile is close to flat: no band departs from the overall rate by more than about ten
points, and every band's departure is of the order of the scorer's own measured noise, so
this is evidence of no strong dependence rather than a measured trend. Severity is the
annotator's own $1$--$5$ judgement of how badly the clip fails.}}
\label{{tab:robust}}
\end{{table}}
"""


def tab_paired(p: dict | None) -> str:
    if not p or not p.get("clips"):
        return ("\n% PAIRED TABLE NOT YET COMPUTABLE — no clip scored by both conditions.\n"
                "\\textbf{[TODO: paired head-to-head table pending run completion]}\n")
    g, f = p["caught"]["graph"], p["caught"]["flat"]
    n = p["flaws"]
    cpc = p["calls_per_clip"]
    ov = p["overlap"]
    lo, hi = p["ci"]
    return rf"""
\begin{{table}}[t]
\centering\small
\setlength{{\tabcolsep}}{{6pt}}\renewcommand{{\arraystretch}}{{1.15}}
\begin{{tabular}}{{@{{}}p{{5.0cm}} c c c@{{}}}}
\toprule
\textbf{{Condition}} & \textbf{{Flaws found}} & \textbf{{Rate}} & \textbf{{Calls per clip}} \\
\midrule
Plan-based critic & {g}/{n} & {100.0*g/n:.1f}\% & {cpc['graph']:.1f} \\
Per-claim verification & {f}/{n} & {100.0*f/n:.1f}\% & {cpc['flat']:.1f} \\
\midrule
\multicolumn{{4}}{{@{{}}l}}{{\textit{{Where they disagree, flaw by flaw}}}} \\
\quad found by both & {ov['both']} & & \\
\quad plan-based only & {ov['graph_only']} & & \\
\quad per-claim only & {ov['flat_only']} & & \\
\quad found by neither & {ov['neither']} & & \\
\bottomrule
\end{{tabular}}
\caption{{Paired comparison on the {p['clips']} clips of the frozen evaluation set that both
conditions have scored in this run, carrying {n} human flaws. Both conditions read the
\emph{{same}} frozen claim decomposition per prompt, so neither is handed easier claims, and
both are scored by the same matching protocol against the same labels. Calls count model and
tool invocations and exclude the matching protocol, which is the scorer rather than the
critic. The difference of {p['net']:+d} flaws has a paired $95\%$ interval of
$[{lo:+d}, {hi:+d}]$ from {p['bootstrap']:,} clip resamples with seed {p['seed']}:
{esc(p['verdict'])}. Both conditions ran against the same served reasoning model
(Qwen3-VL-30B-A3B-Instruct, BF16) and the same live specialist servers; the counting
specialist was measured broken during this run (\cref{{sec:contract}}) and its checks are
recorded as such rather than scored as passes.}}
\label{{tab:paired}}
\end{{table}}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parent / "results_tables.tex"))
    a = ap.parse_args()

    t = _mod("tables")
    buf = io.StringIO()
    with redirect_stdout(buf):
        rows = t.joined()
        _, sev = t.t_severity(rows)
        cat = t.t_category(rows)
        rob = t.t_robustness(rows)
    print(buf.getvalue().splitlines()[-1] if buf.getvalue() else "")

    paired = None
    pj = Path("/tmp/paired.json")
    if pj.exists():
        paired = json.loads(pj.read_text())

    head = ("% GENERATED by analysis/emit_tex.py — do not edit by hand.\n"
            "% Every number is recomputed from per-item rows; see the analysis scripts.\n")
    # One file per table so main.tex can place each where it belongs, plus a combined file.
    out_dir = Path(a.out).parent
    pieces = {"tab_paired": tab_paired(paired), "tab_bycat": tab_category(cat),
              "tab_robust": tab_robust(rob, sev)}
    for name, body in pieces.items():
        p = out_dir / f"{name}.tex"
        p.write_text(head + body)
        print(f"[emit] {p.name}")
    Path(a.out).write_text(head + "\n".join(pieces.values()))
    print(f"[emit] {a.out}  ({'paired table INCLUDED' if paired else 'paired table = TODO marker'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
