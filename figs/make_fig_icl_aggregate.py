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
# fall back to the exported data bundle when the original run tree is absent
_BUNDLE = Path("/Users/jigu/work/reports/techreport_figure_data/fig17_icl_aggregate")
if not (ART / "coverage_curve_k0.json").exists() and _BUNDLE.exists():
    ART = _BUNDLE
    GOLD = _BUNDLE / "gold_full_v1.json"
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


def use_source_serif() -> None:
    """Match the report body face (serif), like the other data figures."""
    import glob
    roots = [
        "/usr/local/texlive/2024/texmf-dist/fonts/opentype/adobe/sourceserifpro",
        str(Path.home() / "Library/Caches/Tectonic/bundles/data/*"),
        str(Path.home() / "Library/Fonts"), "/Library/Fonts",
        "/System/Library/Fonts/Supplemental",
    ]
    for root in roots:
        for base in glob.glob(root):
            for f in glob.glob(str(Path(base) / "SourceSerif*.otf")) + \
                     glob.glob(str(Path(base) / "SourceSerif*.ttf")):
                try:
                    font_manager.fontManager.addfont(f)
                except Exception:
                    pass
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Source Serif 4", "Source Serif Pro", "DejaVu Serif"],
    })


def main() -> int:
    use_source_serif()
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

    # Half-column, near-square. Even categorical x-spacing so the one thing this
    # figure has to show---the jump from untaught to taught, then a dead-flat
    # plateau across dose---is not squeezed into the leftmost fifth of a wide
    # axis. The teaching gain gets a full step and its own arrow; the four taught
    # doses collapse into a thin band whose height is the whole "dose effect".
    RED = "#D64525"  # a legible ink-red for the headline gain (adobered is too pale at this weight)
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    xc = list(range(len(pts)))                     # 0..4, evenly spaced conditions
    ys = [p[2] for p in pts]
    taught = pts[1:]
    plo, phi = min(p[2] for p in taught), max(p[2] for p in taught)  # 374..377
    gain = ys[1] - base                            # +35 at the first taught dose

    # plateau band across the taught doses: its (tiny) height is the dose effect
    ax.fill_between([xc[1] - 0.34, xc[-1] + 0.34], plo, phi, color=LIGHT, zorder=0)
    ax.plot([xc[0], xc[-1]], [base, base], color=GRAY, lw=HAIRLINE, ls=(0, (1.4, 2.2)), zorder=1)

    # whiskers: light and thin so they read as context, not as the subject
    for i, (x, L, y, lo, hi, gd, ls) in enumerate(pts[1:], start=1):
        ax.plot([xc[i], xc[i]], [lo, hi], color=GRAY, lw=0.9, alpha=0.55, zorder=2,
                solid_capstyle="round")
        for e in (lo, hi):
            ax.plot([xc[i] - 0.08, xc[i] + 0.08], [e, e], color=GRAY, lw=0.9, alpha=0.55, zorder=2)

    ax.plot(xc, ys, color=INK, lw=1.5, zorder=3)
    ax.plot(xc, ys, "o", ms=5.6, mfc="white", mec=INK, mew=1.5, zorder=5)

    # headline: the teaching gain, drawn as an arrow from baseline up to the plateau
    xa = 0.5
    ax.annotate("", xy=(xa, plo - 0.8), xytext=(xa, base + 0.8),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.9,
                                shrinkA=0, shrinkB=0, mutation_scale=13), zorder=6)
    ax.text(xa + 0.14, (base + plo) / 2, f"+{gain} flaws\n(+{100 * gain / n:.1f} pts)",
            color=RED, fontsize=8.2, ha="left", va="center", fontweight="bold", linespacing=1.1)

    ax.annotate("untaught  %d (%.1f%%)" % (base, 100 * base / n), (xc[0], base),
                textcoords="offset points", xytext=(3, -5), fontsize=8.0, color=INK,
                ha="left", va="top")
    ax.text(xc[2] + 0.5, phi + 1.2, "taught: %d–%d flaws (≈%.1f%%), flat across dose ($p \\geq 0.8$)"
            % (plo, phi, 100 * ys[-1] / n), fontsize=8.0, color=INK, ha="center", va="bottom")

    ax.set_xlim(-0.42, 4.42)
    ax.set_ylim(334, 392)
    yt = list(range(340, 381, 10))
    ax.set_xticks(xc)
    ax.set_xticklabels([str(p[0]) for p in pts])
    ax.set_yticks(yt)
    ax.set_xlabel("experience clips the lessons were distilled from", fontsize=8.4, color=GRAY)
    ax.set_ylabel(f"flaws found (of {n})", fontsize=8.4, color=GRAY)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(HAIRLINE)
        ax.spines[s].set_color(GRAY)
    ax.spines["left"].set_bounds(yt[0], yt[-1])
    ax.spines["bottom"].set_bounds(xc[0], xc[-1])
    ax.tick_params(labelsize=8.0, colors=GRAY, width=HAIRLINE, length=2.5)
    fig.tight_layout(pad=0.4)
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
