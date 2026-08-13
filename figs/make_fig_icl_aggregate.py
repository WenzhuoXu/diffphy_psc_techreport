#!/usr/bin/env python3
"""The aggregate ICL curve: flaws found against how much experience the lessons were distilled from.

Five conditions over one flaw set. The whisker on each taught point is the paired 95% interval on
its change from the untaught condition, since the conditions score the same flaws.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

R = Path.home() / "diffphy_exp013"
ART = R / "artifacts/runs/exp013/exp029"
GOLD = R / "artifacts/runs/exp013/gold_v1/gold_full_v1.json"
OUT = Path(__file__).resolve().parent

INK, GRAY, LIGHT = "#1A1A1A", "#6E6E6E", "#D8D8D8"
HAIRLINE = 0.6

# lessons -> (experience clips, coverage file, rules file)
DOSE = {0: (0, "coverage_curve_k0.json", "curve_rules_k0.json"),
        8: (25, "coverage_curve_k991.json", "curve_rules_k991.json"),
        13: (50, "coverage_curve_k992.json", "curve_rules_k992.json"),
        19: (75, "coverage_curve_k993.json", "curve_rules_k993.json"),
        33: (117, "coverage_curve_k997.json", "curve_rules_k997.json")}


def paired_ci(gained: int, lost: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval on the paired change, in flaws."""
    d = gained + lost
    if not d:
        return 0.0, 0.0
    ph = gained / d
    den = 1 + z * z / d
    c = (ph + z * z / (2 * d)) / den
    h = z * ((ph * (1 - ph) / d + z * z / (4 * d * d)) ** 0.5) / den
    lo, hi = max(0.0, c - h), min(1.0, c + h)
    return (2 * lo - 1) * d, (2 * hi - 1) * d


def use_source_sans() -> None:
    for p in ("/System/Library/Fonts/Supplemental/", "/Library/Fonts/"):
        for n in ("SourceSans3-Regular.otf", "SourceSansPro-Regular.otf"):
            if (Path(p) / n).exists():
                font_manager.fontManager.addfont(str(Path(p) / n))
                plt.rcParams["font.family"] = font_manager.FontProperties(
                    fname=str(Path(p) / n)).get_name()
                return
    plt.rcParams["font.family"] = "DejaVu Sans"


def main() -> int:
    use_source_sans()
    B, IND = {}, {}
    for L, (_, cf, rf) in DOSE.items():
        B[L] = json.load(open(ART / cf))["framework"]["_buckets"]
        IND[L] = set(json.load(open(ART / rf)).get("induce_clips") or [])
    g = json.load(open(GOLD))
    clips = g["clips"] if isinstance(g, dict) else g
    prompts = {c["item_id"]: (c.get("prompt") or "") for c in clips}
    norm = lambda s: hashlib.sha1(" ".join(s.lower().split()).encode()).hexdigest()  # noqa: E731

    # drop every evaluation clip whose prompt any dose's experience pool also states
    ev = {k.split("#")[0] for k in B[0]}
    leak: set[str] = set()
    for L in (8, 13, 19, 33):
        ih = {norm(prompts[c]) for c in IND[L] if c in prompts}
        leak |= (ev & IND[L]) | {c for c in ev if norm(prompts.get(c, "")) in ih}
    keys = [k for k in sorted(B[0]) if k.split("#")[0] not in leak]
    n = len(keys)
    cov = lambda L, k: B[L][k] in ("caught", "partial")  # noqa: E731

    base = sum(cov(0, k) for k in keys)
    pts = []
    for L, (x, _, _) in DOSE.items():
        c = sum(cov(L, k) for k in keys)
        gd = sum(1 for k in keys if not cov(0, k) and cov(L, k))
        ls = sum(1 for k in keys if cov(0, k) and not cov(L, k))
        lo, hi = paired_ci(gd, ls) if L else (0.0, 0.0)
        pts.append((x, L, c, base + lo, base + hi, gd, ls))

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    xs = [p[0] for p in pts]
    ys = [p[2] for p in pts]
    cap = max(p[4] for p in pts)  # highest whisker top: the taught labels sit in a row above it

    ax.plot([xs[0], xs[-1]], [base, base], color=GRAY, lw=HAIRLINE, ls=(0, (1.4, 2.2)), zorder=1)
    for x, L, y, lo, hi, gd, ls in pts[1:]:
        ax.plot([x, x], [lo, hi], color=GRAY, lw=1.1, zorder=2, solid_capstyle="round")
        for e in (lo, hi):
            ax.plot([x - 1.8, x + 1.8], [e, e], color=GRAY, lw=1.1, zorder=2)
    ax.plot(xs[1:], ys[1:], color=INK, lw=1.5, zorder=3)
    ax.plot(xs[:2], ys[:2], color=INK, lw=1.5, zorder=3)
    ax.plot(xs, ys, "o", ms=6.2, mfc="white", mec=INK, mew=1.5, zorder=5)

    for x, L, y, lo, hi, gd, ls in pts:
        txt = f"{'untaught' if L == 0 else f'{L} lessons'}\n{y}  ({100 * y / n:.1f}%)"
        if L:
            ax.annotate(txt, (x, cap), textcoords="offset points", xytext=(0, 7),
                        fontsize=8.3, color=INK, ha="center", va="bottom", linespacing=1.35)
        else:
            ax.annotate(txt, (x, base), textcoords="offset points", xytext=(9, -7),
                        fontsize=8.3, color=INK, ha="left", va="top", linespacing=1.35)

    ax.set_xlim(-7, 130)
    ax.set_ylim(base - 13, cap + 13)
    yt = list(range(int(base), int(cap) + 1, 10))
    ax.set_xticks(xs)
    ax.set_yticks(yt)
    ax.set_xlabel("experience clips the lessons were distilled from", fontsize=8.6, color=GRAY)
    ax.set_ylabel(f"flaws found (of {n})", fontsize=8.6, color=GRAY)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(HAIRLINE)
        ax.spines[s].set_color(GRAY)
    ax.spines["left"].set_bounds(yt[0], yt[-1])
    ax.spines["bottom"].set_bounds(xs[0], xs[-1])
    ax.tick_params(labelsize=8.1, colors=GRAY, width=HAIRLINE, length=2.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_icl_aggregate.{ext}", dpi=220)
        print(f"[write] {OUT / f'fig_icl_aggregate.{ext}'}")
    print(f"  {n} flaws on {len(ev) - len(leak)} clips ({len(leak)} of {len(ev)} dropped for leakage)")
    for x, L, y, lo, hi, gd, ls in pts:
        print(f"  {L:>3} lessons / {x:>3} clips: {y:>3} of {n} = {100*y/n:5.1f}%   "
              f"change {y-base:+3d}  [{lo-base:+.1f}, {hi-base:+.1f}]   {gd} gained / {ls} lost")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
