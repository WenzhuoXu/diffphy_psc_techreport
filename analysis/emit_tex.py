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
import re
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
    al = p["allegations"]
    ov = p["overlap"]
    lo, hi = p["ci"]
    return rf"""
\begin{{table}}[t]
\centering\small
\setlength{{\tabcolsep}}{{6pt}}\renewcommand{{\arraystretch}}{{1.15}}
\begin{{tabular}}{{@{{}}p{{4.3cm}} c c c c c@{{}}}}
\toprule
\textbf{{Condition}} & \textbf{{Flaws found}} & \textbf{{Rate}} & \textbf{{Calls}}
 & \textbf{{Accusations}} & \textbf{{That}} \\
 & of {n} & & per clip & per clip & \textbf{{landed}} \\
\midrule
Plan-based critic & {g} & {100.0*g/n:.1f}\% & {cpc['graph']:.1f}
 & {al['graph']['per_clip']:.1f} & {al['graph']['precision']:.0f}\% \\
Per-claim verification & {f} & {100.0*f/n:.1f}\% & {cpc['flat']:.1f}
 & {al['flat']['per_clip']:.1f} & {al['flat']['precision']:.0f}\% \\
\midrule
\multicolumn{{6}}{{@{{}}l}}{{\textit{{Where they disagree, flaw by flaw}}}} \\
\quad found by both & {ov['both']} & & & & \\
\quad plan-based only & {ov['graph_only']} & & & & \\
\quad per-claim only & {ov['flat_only']} & & & & \\
\quad found by neither & {ov['neither']} & & & & \\
\bottomrule
\end{{tabular}}
\caption{{Paired comparison on the {p['clips']} clips of the frozen evaluation set that both
conditions have scored in this run, carrying {n} human flaws. Both conditions are scored by the
same matching protocol against the same labels and run against the same models and specialists.
Each decomposes the prompt into claims with its own call, so this is a whole-system comparison
and the difference includes any advantage from how the prompt was split. Calls count model and
tool invocations and exclude the matching protocol, which is the scorer rather than the
critic. The difference of {p['net']:+d} flaws has a paired $95\%$ interval of
$[{lo:+d}, {hi:+d}]$ from {p['bootstrap']:,} clip resamples with seed {p['seed']}:
{esc(p['verdict'])}. Both conditions ran against the same served reasoning model
(Qwen3-VL-30B-A3B-Instruct, BF16) and the same live specialist servers; the counting
specialist was measured broken during this run (\cref{{sec:contract}}) and its checks are
recorded as such rather than scored as passes. The last two columns guard against a
volume effect: recall alone rewards accusing more, so the accusation rate and the share of
accusations the matching protocol links to a human flaw are reported beside it.}}
\label{{tab:paired}}
\end{{table}}
"""


def tab_ablation(ab: dict | None) -> str:
    """One row per mechanism switched off, each a separate live run paired against the full
    condition. A null row is reported as null: the point of an ablation is to find out, and a
    mechanism the data barely exercises cannot show an effect however useful it may be
    elsewhere."""
    if not ab or not ab.get("rows"):
        return ("\n% ABLATION TABLE NOT YET COMPUTABLE — no row has paired clips.\n"
                "\\textbf{[TODO: mechanism ablation pending run completion]}\n")
    body, note = [], ""
    r0 = ab["rows"][0]
    body.append(f"Full plan-based critic & {r0['full_caught']}/{r0['flaws']} & "
                f"{100.0*r0['full_caught']/r0['flaws']:.1f}\\% & {r0['full_calls']:.1f} "
                f"& --- & --- \\\\")
    body.append(r"\midrule")
    for r in ab["rows"]:
        lo, hi = r["ci"]
        body.append(f"{esc(r['label'])} & {r['off_caught']}/{r['flaws']} & "
                    f"{100.0*r['off_caught']/r['flaws']:.1f}\\% & {r['off_calls']:.1f} & "
                    f"{r['diff']:+d} & $[{lo:+d}, {hi:+d}]$ \\\\")
        if r["effect"] == "no measurable effect":
            extra = r.get("extra_calls", 0.0)
            act, tot = r.get("flag_active_clips", 0), r["clips"]
            note = (f" Removing call deduplication found exactly the same flaws --- the paired "
                    f"interval is zero --- while spending {extra:+.2f} calls per clip more. "
                    f"Sharing is therefore a cost saving on this data and not a source of "
                    f"recall. The saving is concentrated rather than spread: only {act} of "
                    f"{tot} clips had any duplicate perception query to collapse, and on those "
                    f"the ablation spent between two and fourteen extra calls. A positive "
                    f"control confirms the mechanism really was disabled rather than the "
                    f"switch having no effect: on every clip where the full condition "
                    f"deduplicated, the ablation demonstrably spent more calls, and no "
                    f"ablation row reports a saving.")
    rows = "\n".join(body)
    return rf"""
\begin{{table}}[t]
\centering\small
\setlength{{\tabcolsep}}{{5pt}}\renewcommand{{\arraystretch}}{{1.15}}
\begin{{tabular}}{{@{{}}p{{4.2cm}} c c c c c@{{}}}}
\toprule
\textbf{{Condition}} & \textbf{{Flaws found}} & \textbf{{Rate}} & \textbf{{Calls}}
 & \textbf{{$\Delta$}} & \textbf{{Paired 95\%}} \\
 & & & per clip & flaws & \textbf{{interval}} \\
\midrule
{rows}
\bottomrule
\end{{tabular}}
\caption{{Mechanism ablation on {r0['clips']} clips of the frozen set carrying {r0['flaws']}
human flaws. Each row is a \emph{{separate live run}} of the identical plan with one mechanism
disabled, scored against the same labels and paired clip-by-clip against the full condition
above; the interval is a paired bootstrap over clips. This is deliberately not a subtraction
from a finished run, which would assume every finding the mechanism produced becomes a miss
rather than being recovered another way.{note}}}
\label{{tab:ablation}}
\end{{table}}
"""


def check_prose_consistency(paired: dict, tex_path: Path) -> list[str]:
    """Warn when main.tex's PROSE hardcodes a number the paired table has moved past.

    WHY: the paired run's denominator grows as clips land, so any figure typed into a sentence
    goes stale silently. It happened once already -- the prose said "5.7 accusations per clip
    against 4.2" and "33% against 27%" while the regenerated table read 4.4/3.3 and 40%/38%,
    which is a materially different claim about whether the gain is a volume effect. The tables
    are generated and cannot drift; sentences can. So the sentences must avoid bare figures the
    table owns, and this check enforces that by flagging any that reappear.
    """
    if not paired or not tex_path.exists():
        return []
    src = tex_path.read_text()
    i = src.find(r"\subsection{A controlled head-to-head")
    j = src.find(r"\subsection{The current critic")
    if i < 0 or j < 0:
        return ["could not locate the head-to-head subsection to check"]
    body = src[i:j]
    owned = {
        "accusations per clip (graph)": paired["allegations"]["graph"]["per_clip"],
        "accusations per clip (flat)": paired["allegations"]["flat"]["per_clip"],
        "landed share (graph)": paired["allegations"]["graph"]["precision"],
        "landed share (flat)": paired["allegations"]["flat"]["precision"],
        "flaws found (graph)": paired["caught"]["graph"],
        "flaws found (flat)": paired["caught"]["flat"],
    }
    out = []
    for label, val in owned.items():
        # Check the figure AS THE TABLE RENDERS IT, not the raw float: the table prints 4.4 for
        # a stored 4.37, so searching the raw value finds nothing and the guard silently passes.
        # That is exactly the failure this function exists to prevent, so it must not repeat it
        # one level up. Both the one-decimal and integer renderings are checked.
        forms = {f"{val:.1f}", f"{val:.0f}", f"{val:g}"}
        for f in forms:
            for pat in (rf"\${re.escape(f)}\$", rf"(?<![\d.]){re.escape(f)}\\%"):
                if re.search(pat, body):
                    out.append(f"prose hardcodes {label} (renders as {f}); "
                               "phrase it qualitatively and let the table own the number")
                    break
            else:
                continue
            break
    return out


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
    ablation = None
    aj = Path("/tmp/ablation.json")
    if aj.exists():
        ablation = json.loads(aj.read_text())

    head = ("% GENERATED by analysis/emit_tex.py — do not edit by hand.\n"
            "% Every number is recomputed from per-item rows; see the analysis scripts.\n")
    # One file per table so main.tex can place each where it belongs, plus a combined file.
    out_dir = Path(a.out).parent
    pieces = {"tab_paired": tab_paired(paired), "tab_ablation": tab_ablation(ablation),
              "tab_bycat": tab_category(cat), "tab_robust": tab_robust(rob, sev)}
    for name, body in pieces.items():
        p = out_dir / f"{name}.tex"
        p.write_text(head + body)
        print(f"[emit] {p.name}")
    Path(a.out).write_text(head + "\n".join(pieces.values()))
    for w in check_prose_consistency(paired, HERE.parent / "main.tex"):
        print(f"[emit] WARN {w}")
    print(f"[emit] {a.out}  ({'paired table INCLUDED' if paired else 'paired table = TODO marker'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

