#!/usr/bin/env python3
"""Emit tab_icl_category.tex -- per-category effect of the distilled lessons, five doses.

Supersedes the pair of overlapping tables this section used to carry (a three-dose category table
and a separate untaught-vs-taught one): they reported the same flaws twice.

The five conditions score the SAME flaws, so the comparison is paired: the interval on the change
comes from the flaws that switched, not from two independent rates.

  python emit_icl_category_tex.py --out ../tab_icl_category.tex
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from math import comb
from pathlib import Path

R = Path.home() / "diffphy_exp013"
ART = R / "artifacts/runs/exp013/exp029"
GOLD = R / "artifacts/runs/exp013/gold_v1/gold_full_v1.json"

# lessons -> (coverage file, rules file, experience clips the lessons were distilled from)
DOSE = {0: ("coverage_curve_k0.json", "curve_rules_k0.json", 0),
        8: ("coverage_curve_k991.json", "curve_rules_k991.json", 25),
        13: ("coverage_curve_k992.json", "curve_rules_k992.json", 50),
        19: ("coverage_curve_k993.json", "curve_rules_k993.json", 75),
        33: ("coverage_curve_k997.json", "curve_rules_k997.json", 117)}

MIN_FLAWS = 12  # categories below this are pooled into the caption, not shown as rows

PRETTY = {"action": "action or event occurs", "text_ocr": "on-screen text",
          "object": "object identity", "spatial": "spatial relation", "attribute": "attribute",
          "count": "how many", "order_timing": "order and timing",
          "camera_style": "camera and style", "physics_motion": "physical motion",
          "other": "other"}

# which flaw category a lesson family names, by the words in the family's own name
FAMILY_TARGET = {"cardinality": "count", "aggregate": "count", "spatial": "spatial",
                 "text": "text_ocr", "temporal": "order_timing", "identity": "object",
                 "subject": "object", "whole-part": "object", "visual attribute": "attribute",
                 "specificity": "attribute"}


def target_of(family: str) -> str:
    for k, v in FAMILY_TARGET.items():
        if k in family:
            return v
    return "action"


def sign_p(a: int, b: int) -> float:
    """Two-sided exact sign test over the flaws that switched."""
    n = a + b
    if not n:
        return 1.0
    return min(1.0, sum(comb(n, i) for i in range(0, min(a, b) + 1)) * 2 / 2 ** n)


def paired_ci(gained: int, lost: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval on the paired change, in percentage points of n."""
    d = gained + lost
    if not d or not n:
        return 0.0, 0.0
    ph = gained / d
    den = 1 + z * z / d
    c = (ph + z * z / (2 * d)) / den
    h = z * ((ph * (1 - ph) / d + z * z / (4 * d * d)) ** 0.5) / den
    lo, hi = max(0.0, c - h), min(1.0, c + h)
    return 100.0 * (2 * lo - 1) * d / n, 100.0 * (2 * hi - 1) * d / n


def load():
    """Buckets per dose, the gold flaw records, and the prompt-leaked clips to drop."""
    B, IND = {}, {}
    for L, (cf, rf, _) in DOSE.items():
        B[L] = json.load(open(ART / cf))["framework"]["_buckets"]
        IND[L] = set(json.load(open(ART / rf)).get("induce_clips") or [])
    assert all(set(B[L]) == set(B[0]) for L in B), "the doses scored different flaw sets"

    g = json.load(open(GOLD))
    clips = g["clips"] if isinstance(g, dict) else g
    prompts = {c["item_id"]: (c.get("prompt") or "") for c in clips}
    meta = {f"{c['item_id']}#{i}": f for c in clips for i, f in enumerate(c.get("flaws") or [])}

    # An evaluation clip is dropped if ANY dose's experience pool contains it, or contains a clip
    # stating the same prompt. Filtering on the union keeps one flaw set across all five doses,
    # which is what makes the comparison paired.
    norm = lambda s: hashlib.sha1(" ".join(s.lower().split()).encode()).hexdigest()  # noqa: E731
    ev = {k.split("#")[0] for k in B[0]}
    leak: set[str] = set()
    for L in (8, 13, 19, 33):
        ih = {norm(prompts[c]) for c in IND[L] if c in prompts}
        leak |= (ev & IND[L]) | {c for c in ev if norm(prompts.get(c, "")) in ih}
    keys = [k for k in sorted(B[0]) if k.split("#")[0] not in leak]
    return B, meta, keys, len(ev), len(leak)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    B, meta, keys, n_ev, n_leak = load()
    cov = lambda L, k: B[L][k] in ("caught", "partial")  # noqa: E731

    # lesson families naming each category, counted over the full 33
    fams = [l["family"] for l in json.load(open(ART / DOSE[33][1]))["open_lessons"]]
    n_fam = collections.Counter(target_of(f) for f in fams)

    per = collections.defaultdict(lambda: collections.defaultdict(int))
    for k in keys:
        d = per[str(meta[k].get("category"))]
        d["n"] += 1
        for L in DOSE:
            d[L] += cov(L, k)
        if not cov(0, k) and cov(33, k):
            d["g"] += 1
        if cov(0, k) and not cov(33, k):
            d["l"] += 1

    shown = [c for c in per if per[c]["n"] >= MIN_FLAWS]
    shown.sort(key=lambda c: -(per[c][33] - per[c][0]))
    hid = [c for c in per if per[c]["n"] < MIN_FLAWS]
    n_hid = sum(per[c]["n"] for c in hid)

    n = len(keys)
    tot = {L: sum(cov(L, k) for k in keys) for L in DOSE}
    G = sum(1 for k in keys if not cov(0, k) and cov(33, k))
    Ls = sum(1 for k in keys if cov(0, k) and not cov(33, k))

    out = ["% GENERATED by techreport/figs/emit_icl_category_tex.py -- do not edit by hand.",
           "% Paired comparison: the five conditions score the same flaws.",
           "",
           r"\begin{table}[t]",
           r"\centering\small",
           r"\renewcommand{\arraystretch}{1.2}",
           r"\begin{tabularx}{\textwidth}{@{}L c c c c c c c c c@{}}",
           r"\toprule",
           r" & & \multicolumn{5}{c}{\textbf{Flaws found, with $\ell$ lessons}} & & & \\",
           r"\cmidrule(lr){3-7}",
           r"\textbf{Flaw category} & \textbf{Flaws} & $\ell{=}0$ & $8$ & $13$ & $19$ & $33$ & "
           r"\textbf{Change} & \textbf{95\% interval (pts)} & \textbf{Families} \\",
           r"\midrule"]

    def row(label: str, d, n_row: int, gd: int, ls: int, fam: int | None, bold: bool) -> str:
        ch = 100.0 * (d[33] - d[0]) / n_row
        lo, hi = paired_ci(gd, ls, n_row)
        cells = [f"{n_row}"] + [f"{d[L]}" for L in (0, 8, 13, 19, 33)]
        cells += [f"{d[33]-d[0]:+d} ({ch:+.0f})", f"$[{lo:+.0f}, {hi:+.0f}]$",
                  "" if fam is None else f"{fam}"]
        if bold:
            label = r"\textbf{" + label + "}"
            cells = [r"\textbf{" + c + "}" if c else c for c in cells]
        return label + " & " + " & ".join(cells) + r" \\"

    for c in shown:
        d = per[c]
        lo, hi = paired_ci(d["g"], d["l"], d["n"])
        out.append(row(PRETTY.get(c, c), d, d["n"], d["g"], d["l"], n_fam.get(c, 0),
                       bold=lo > 0 or hi < 0))
    out.append(r"\midrule")
    out.append(row("All flaws", {L: tot[L] for L in DOSE}, n, G, Ls, len(fams), bold=True))
    out += [r"\bottomrule",
            r"\end{tabularx}",
            r"\caption{Effect of the distilled lessons on each flaw category, over the same "
            f"${n}$ held-out flaws in every condition. A flaw is \\emph{{found}} when the critic's "
            r"findings cover it wholly or in part, as in \cref{tab:headline}. The five middle "
            r"columns count that category's flaws found with no lessons and with $8$, $13$, $19$ "
            f"and $33$ lessons, distilled from $25$, $50$, $75$ and $117$ experience clips; "
            r"\textbf{Change} is the untaught-to-$33$ difference in flaws, with percentage points "
            r"in brackets, since several categories carry few enough flaws that a rate alone would "
            r"overstate the movement. Because the conditions score the same flaws the comparison "
            r"is paired: the interval on the change is computed from the flaws whose found status "
            r"switched, and is given in percentage points of that category's flaws. \textbf{Bold} "
            r"marks the rows whose interval excludes zero. \textbf{Families} counts, over all $33$ "
            r"lessons, those whose family name refers to that category, matching on the words in "
            f"the name. The {', '.join(PRETTY.get(c, c) for c in sorted(hid))} "
            f"categor{'y' if len(hid) == 1 else 'ies'}, containing ${n_hid}$ flaws in all, "
            f"{'is' if len(hid) == 1 else 'are'} omitted from the category rows; "
            r"the \emph{All flaws} row includes those flaws.}",
            r"\label{tab:iclcategory}",
            r"\end{table}",
            ""]
    Path(a.out).write_text("\n".join(out))

    print(f"-> {a.out}")
    print(f"   {n_ev} evaluation clips, {n_leak} dropped for prompt leakage, {n} flaws scored")
    print(f"   {'category':26s}{'n':>4}" + "".join(f"{L:>5}" for L in DOSE) +
          f"{'chg':>6}{'g/l':>9}{'95% CI':>16}{'p':>9}{'fam':>5}")
    for c in shown + hid:
        d = per[c]
        lo, hi = paired_ci(d["g"], d["l"], d["n"])
        print(f"   {PRETTY.get(c,c):26s}{d['n']:>4}" + "".join(f"{d[L]:>5}" for L in DOSE) +
              f"{d[33]-d[0]:>+6}{f'{d[chr(103)]}/{d[chr(108)]}':>9}{f'[{lo:+.0f},{hi:+.0f}]':>16}"
              f"{sign_p(d['g'], d['l']):>9.3f}{n_fam.get(c,0):>5}" +
              ("   (omitted)" if c in hid else ""))
    lo, hi = paired_ci(G, Ls, n)
    print(f"   {'ALL':26s}{n:>4}" + "".join(f"{tot[L]:>5}" for L in DOSE) +
          f"{tot[33]-tot[0]:>+6}{f'{G}/{Ls}':>9}{f'[{lo:+.1f},{hi:+.1f}]':>16}"
          f"{sign_p(G, Ls):>9.5f}{len(fams):>5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
