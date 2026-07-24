#!/usr/bin/env python3
"""Build the three benchmark figures for the technical report.

Every number plotted here is computed from the real annotation files listed in
SOURCES below; nothing is hard-coded from prose. The script prints every
quantity it computes so the figures can be audited against the data.

Outputs (PDF for LaTeX + PNG at 200 dpi for review), written next to this file:
    fig_benchmark_composition.{pdf,png}
    fig_benchmark_anatomy.{pdf,png}
    fig_benchmark_distributions.{pdf,png}

Run:
    python3 /Users/wenzhuox/diffphy_psc/techreport/figs/make_fig_benchmark.py

Idempotent: re-running overwrites the same six files and re-prints the same
numbers.
"""
from __future__ import annotations

import glob
import json
import statistics
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.patches import Ellipse, Rectangle  # noqa: E402

# --------------------------------------------------------------------------- #
# sources (absolute; read-only)
# --------------------------------------------------------------------------- #
EXP013 = Path("/Users/wenzhuox/diffphy_exp013/artifacts/runs/exp013")
SOURCES = {
    "raw_1500": EXP013 / "exp020_corpus/adobe_pilot_data_raw_client_1500.json",
    "pool_606": EXP013 / "pilot_manifests/manifest.json",
    "flaw_class": EXP013 / "gold_v1/flaw_class.jsonl",
    "gold_full": EXP013 / "gold_v1/gold_full_v1.json",
    "gold_core": EXP013 / "gold_v1/gold_core_v1.json",
    "train_pool": EXP013 / "exp020_corpus/train_pool.json",
}
OUT = Path("/Users/wenzhuox/diffphy_psc/techreport/figs")

# --------------------------------------------------------------------------- #
# report palette + typography (must match adobe-techreport.sty)
# --------------------------------------------------------------------------- #
ADOBE_RED = "#FA0F00"
INK = "#1A1A1A"
GRAY = "#6E6E6E"
FILL_MID = "#E5E5E5"
FILL_LIGHT = "#F5F5F5"

FONT_DIRS = [
    "/Users/wenzhuox/Library/Caches/Tectonic/bundles/data/*",  # the report's own font copies
    str(Path.home() / "Library/Fonts"),
    "/Library/Fonts",
    "/usr/share/fonts/**",
]


def register_source_fonts() -> dict[str, str]:
    """Register the Source font family used by the report with matplotlib.

    The report compiles with Source Sans Pro / Source Serif Pro / Source Code
    Pro, and tectonic keeps the exact OTFs in its bundle cache, so the figures
    use the identical faces rather than a lookalike or the matplotlib default.
    """
    found: dict[str, str] = {}
    patterns = ("SourceSansPro-*.otf", "SourceSerifPro-*.otf", "SourceCodePro-*.otf")
    for root in FONT_DIRS:
        for pat in patterns:
            for path in glob.glob(f"{root}/{pat}", recursive=True):
                name = Path(path).name
                found.setdefault(name, path)
    for path in found.values():
        try:
            fm.fontManager.addfont(path)
        except Exception:  # pragma: no cover - font parse failure is non-fatal
            pass
    families = {f.name for f in fm.fontManager.ttflist}
    have_sans = "Source Sans Pro" in families
    if not have_sans:
        raise SystemExit(
            "Source Sans Pro not found; refusing to fall back to a default face. "
            "Searched: " + ", ".join(FONT_DIRS)
        )
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Source Sans 3", "Source Sans Pro", "sans-serif"],
            "font.serif": ["Source Serif 4", "Source Serif Pro", "serif"],
            "font.monospace": ["Source Code Pro", "monospace"],
            "font.size": 9,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": GRAY,
            "axes.linewidth": 0.7,
            "xtick.color": GRAY,
            "ytick.color": GRAY,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.grid": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return found


SANS_SEMI = None  # filled in by register_source_fonts / _semibold()


def _semibold(size: float):
    """FontProperties for Source Sans Pro Semibold (the report's heading weight)."""
    global SANS_SEMI
    if SANS_SEMI is None:
        hits = []
        for root in FONT_DIRS:
            hits += glob.glob(f"{root}/SourceSansPro-Semibold.otf", recursive=True)
        SANS_SEMI = hits[0] if hits else ""
    if SANS_SEMI:
        return fm.FontProperties(fname=SANS_SEMI, size=size)
    return fm.FontProperties(family="sans-serif", weight="semibold", size=size)


def _serif(size: float, italic: bool = False):
    hits = []
    want = "SourceSerifPro-RegularIt.otf" if italic else "SourceSerifPro-Regular.otf"
    for root in FONT_DIRS:
        hits += glob.glob(f"{root}/{want}", recursive=True)
    if hits:
        return fm.FontProperties(fname=hits[0], size=size)
    return fm.FontProperties(family="serif", size=size, style="italic" if italic else "normal")


def _serif_bold(size: float):
    hits = []
    for root in FONT_DIRS:
        hits += glob.glob(f"{root}/SourceSerifPro-Semibold.otf", recursive=True)
    if hits:
        return fm.FontProperties(fname=hits[0], size=size)
    return fm.FontProperties(family="serif", size=size, weight="semibold")


# --------------------------------------------------------------------------- #
# data loading + verification
# --------------------------------------------------------------------------- #
def load_all() -> dict:
    raw_list = json.loads(SOURCES["raw_1500"].read_text())
    raw = {x["item_id"]: x for x in raw_list}
    pool = json.loads(SOURCES["pool_606"].read_text())
    flaw_class = [json.loads(ln) for ln in SOURCES["flaw_class"].read_text().splitlines() if ln.strip()]
    gold_full = json.loads(SOURCES["gold_full"].read_text())
    gold_core_doc = json.loads(SOURCES["gold_core"].read_text())
    train_pool = json.loads(SOURCES["train_pool"].read_text())
    return dict(
        raw_list=raw_list,
        raw=raw,
        pool=pool,
        flaw_class=flaw_class,
        gold_full=gold_full,
        gold_core_doc=gold_core_doc,
        gold_core=gold_core_doc["clips"],
        train_pool=train_pool,
    )


def compute_funnel(D: dict) -> list[dict]:
    """Every stage of the evaluation-set derivation, counted from the files."""
    raw_list, pool = D["raw_list"], D["pool"]
    flaw_class, gold_full, core = D["flaw_class"], D["gold_full"], D["gold_core"]

    n_raw_clips = len(raw_list)
    n_raw_flaws = sum(len(x["annotation"]["mismatch_entries"]) for x in raw_list)

    pool_ids = [c["item_id"] for c in pool["clips"]]
    n_pool_clips = len(pool_ids)
    n_pool_flaws = sum(len(D["raw"][i]["annotation"]["mismatch_entries"]) for i in pool_ids)
    # independent cross-check: one classification row per pooled flaw
    assert len(flaw_class) == n_pool_flaws, (len(flaw_class), n_pool_flaws)
    assert len({r["item_id"] for r in flaw_class}) == n_pool_clips

    n_eval_clips = len(gold_full)
    n_eval_flaws = sum(len(c["flaws"]) for c in gold_full)
    # independent cross-check: the frozen set keeps the visually-scoped flaws
    kept = [r for r in flaw_class if r["scope"] in ("visual", "mixed")]
    assert len(kept) == n_eval_flaws, (len(kept), n_eval_flaws)

    n_core_clips = len(core)
    n_core_flaws = sum(len(c["flaws"]) for c in core)

    return [
        dict(
            label="Annotated corpus",
            sub="every clip carries at least one human-flagged flaw",
            clips=n_raw_clips,
            flaws=n_raw_flaws,
        ),
        dict(
            label="Physics-relevant pool",
            sub="the prompt describes motion, or the flagged moment is an action",
            clips=n_pool_clips,
            flaws=n_pool_flaws,
        ),
        dict(
            label="Frozen evaluation set",
            sub="flaws a viewer can see, audio-only complaints removed",
            clips=n_eval_clips,
            flaws=n_eval_flaws,
        ),
        dict(
            label="Routine evaluation core",
            sub="fixed random sample, balanced by severity, flaw count and length",
            clips=n_core_clips,
            flaws=n_core_flaws,
        ),
    ]


def compute_side_counts(D: dict) -> dict:
    """Drop accounting and the disjoint training pool, from the files."""
    pool_stats = D["pool"]["stats"]
    flaw_class = D["flaw_class"]
    scope = Counter(r["scope"] for r in flaw_class)
    core_ids = {c["item_id"] for c in D["gold_core"]}
    eval_ids = {c["item_id"] for c in D["gold_full"]}
    train_ids = {c["item_id"] for c in D["train_pool"]}
    cont = [c for c in D["gold_core"] if c["strata"].get("adv29")]
    return dict(
        dropped_no_video=pool_stats["dropped_no_video"],
        dropped_not_physics=pool_stats["dropped_not_physics"],
        audio_only_flaws=scope["audio"],
        scope_hist=dict(scope),
        continuity_clips=len(cont),
        continuity_flaws=sum(len(c["flaws"]) for c in cont),
        train_clips=len(train_ids),
        train_flaws=sum(len(x["mismatches"]) for x in D["train_pool"]),
        train_overlap_eval=len(train_ids & eval_ids),
        train_overlap_core=len(train_ids & core_ids),
        core_subset_of_eval=core_ids <= eval_ids,
    )


def compute_core_stats(D: dict) -> dict:
    """Per-clip and per-flaw statistics of the 150-clip core.

    A flaw's time span comes from the annotator's start/end frame index in the
    raw corpus, joined to the frozen core record by its flaw position; the join
    is verified exactly (prompt fragment, severity and rationale must all match)
    before any span is used.
    """
    raw, core = D["raw"], D["gold_core"]
    per_clip_flaws = [len(c["flaws"]) for c in core]

    joined = 0
    spans = []  # (seconds, fraction_of_clip)
    no_span = 0
    cat = Counter()
    sev = Counter()
    for c in core:
        entries = raw[c["item_id"]]["annotation"]["mismatch_entries"]
        n_frames = max(1, round(c["fps"] * c["duration_s"]))
        for f in c["flaws"]:
            e = entries[f["m_idx"]]
            frag = " / ".join(e.get("mismatched_prompt") or [])
            assert frag.strip() == f["span"].strip()
            assert e["severity"] == f["severity"] and e["reasoning"] == f["reasoning"]
            joined += 1
            cat[f["category"]] += 1
            sev[f["severity"]] += 1
            s, t = e["start_frame_index"], e["end_frame_index"]
            if s is None or t is None:
                no_span += 1
                continue
            n_span = t - s + 1  # frame indices are inclusive
            spans.append((n_span / c["fps"], min(n_span / n_frames, 1.0)))

    secs = [a for a, _ in spans]
    fracs = [b for _, b in spans]
    durs = [c["duration_s"] for c in core]
    return dict(
        n_clips=len(core),
        n_flaws=sum(per_clip_flaws),
        joined=joined,
        per_clip_hist=dict(sorted(Counter(per_clip_flaws).items())),
        per_clip_mean=statistics.mean(per_clip_flaws),
        per_clip_median=statistics.median(per_clip_flaws),
        per_clip_max=max(per_clip_flaws),
        n_timed=len(spans),
        n_untimed=no_span,
        span_s=secs,
        span_frac=fracs,
        span_s_median=statistics.median(secs),
        span_s_mean=statistics.mean(secs),
        span_s_min=min(secs),
        span_s_max=max(secs),
        span_frac_median=statistics.median(fracs),
        span_frac_mean=statistics.mean(fracs),
        span_frac_min=min(fracs),
        span_frac_max=max(fracs),
        frac_le_25=sum(1 for v in fracs if v <= 0.25),
        frac_le_50=sum(1 for v in fracs if v <= 0.50),
        frac_ge_90=sum(1 for v in fracs if v >= 0.90),
        dur_min=min(durs),
        dur_median=statistics.median(durs),
        dur_max=max(durs),
        cat_hist=dict(cat.most_common()),
        sev_hist=dict(sorted(sev.items())),
    )


# the flaw record drawn in the anatomy figure; chosen for a short self-contained
# prompt, a clearly-worded rationale and a span that sits inside the clip
ANATOMY_ITEM = "2231bee5-b2a1-577f-b120-0840a35091ef"
VIDEO_CACHE = EXP013 / "video_cache"


def pick_anatomy(D: dict, item_id: str = ANATOMY_ITEM) -> dict:
    core = {c["item_id"]: c for c in D["gold_core"]}
    clip = core[item_id]
    entry_list = D["raw"][item_id]["annotation"]["mismatch_entries"]
    flaw = clip["flaws"][0]
    e = entry_list[flaw["m_idx"]]
    frag = " / ".join(e.get("mismatched_prompt") or [])
    assert frag.strip() == flaw["span"].strip()
    assert flaw["span"] in clip["prompt"], "fragment must be quotable from the prompt"
    s, t = e["start_frame_index"], e["end_frame_index"]
    boxes = (e.get("bbox_data") or {}).get("boxes") or {}
    key_frames = sorted({b["frame_index"] for b in boxes.values() if b.get("frame_index") is not None})
    box_by_frame = {}
    for b in boxes.values():
        fi = b.get("frame_index")
        anns = b.get("annotations") or []
        if fi is not None and anns:
            a = anns[0]
            box_by_frame[fi] = (a["x_min"], a["y_min"], a["x_max"], a["y_max"])
    return dict(
        item_id=item_id,
        prompt=clip["prompt"],
        span_text=flaw["span"],
        reasoning=flaw["reasoning"],
        severity=flaw["severity"],
        confidence=e["confidence"],
        fps=clip["fps"],
        duration_s=clip["duration_s"],
        start_frame=s,
        end_frame=t,
        start_s=s / clip["fps"],
        end_s=t / clip["fps"],          # label the last flagged frame itself
        end_s_excl=(t + 1) / clip["fps"],  # inclusive end, kept for the length
        span_s=(t - s + 1) / clip["fps"],
        n_boxed_keyframes=len(key_frames),
        key_frames=key_frames,
        key_frame_s=[k / clip["fps"] for k in key_frames],
        box_by_frame=box_by_frame,
        n_flaws_in_clip=len(clip["flaws"]),
        video=str(VIDEO_CACHE / f"{item_id}.mp4"),
    )


def read_frames(path: str, wanted: list[int]) -> dict[int, "object"]:
    """Pull specific frames out of the real clip (RGB, for matplotlib)."""
    try:
        import cv2
    except ImportError:
        return {}
    if not Path(path).exists():
        return {}
    cap = cv2.VideoCapture(path)
    want = set(wanted)
    out: dict[int, object] = {}
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in want:
            out[i] = fr[:, :, ::-1].copy()
        i += 1
    cap.release()
    return out


# --------------------------------------------------------------------------- #
# (a) composition funnel
# --------------------------------------------------------------------------- #
def fig_composition(stages: list[dict], side: dict) -> None:
    """Four rows, one per stage: name and explanation left, two bar columns right.

    Row geometry is computed in figure coordinates so nothing can collide: each
    stage owns a fixed-height band, the bars sit in the upper part of the band
    and the wrapped between-stage note sits in the gap below it.
    """
    fig = plt.figure(figsize=(6.5, 3.9))
    n = len(stages)

    L = 0.030                       # left margin
    NAME_W = 0.300                  # stage name + explanation column
    GAP = 0.024
    COL1_X = L + NAME_W + GAP
    COL_W = 0.275                   # each bar column
    COL2_X = COL1_X + COL_W + 0.075

    TOP = 0.900                     # baseline of the column captions
    ROW_H = 0.196                   # one stage band
    BAR_H = 0.052                   # bar thickness in figure units

    max_v = {"clips": max(s["clips"] for s in stages), "flaws": max(s["flaws"] for s in stages)}
    # light-to-dark ramp: the set gets smaller and more closely examined
    shades = ["#F5F5F5", "#E5E5E5", GRAY, INK]

    for x0, key, cap in ((COL1_X, "clips", "video clips"), (COL2_X, "flaws", "human-marked flaws")):
        fig.text(x0, TOP, cap, fontproperties=_semibold(8.8), color=GRAY, va="baseline")

    rows = []
    for i, (st, sh) in enumerate(zip(stages, shades)):
        y_bar = TOP - 0.055 - i * ROW_H - BAR_H  # bottom edge of this row's bars
        rows.append(y_bar)
        # stage name and its plain-language explanation, right-aligned to the bars
        fig.text(L + NAME_W, y_bar + BAR_H * 0.62, st["label"], ha="right", va="baseline",
                 fontproperties=_semibold(9.0), color=INK)
        for j, line in enumerate(_wrap(fig, st["sub"], NAME_W, _serif(7.5))):
            fig.text(L + NAME_W, y_bar - 0.014 - j * 0.030, line, ha="right", va="baseline",
                     fontproperties=_serif(7.5), color=GRAY)
        for x0, key in ((COL1_X, "clips"), (COL2_X, "flaws")):
            frac = st[key] / max_v[key]
            w = COL_W * 0.80 * frac
            fig.add_artist(Rectangle((x0, y_bar), w, BAR_H, transform=fig.transFigure,
                                     facecolor=sh, edgecolor="none"))
            fig.text(x0 + w + 0.008, y_bar + BAR_H * 0.30, f"{st[key]:,}", ha="left",
                     va="baseline", fontproperties=_semibold(9.2),
                     color=INK if sh == INK else GRAY)

    # what each narrowing step sets aside -- every count read from the manifests
    drops = [
        f"{side['dropped_no_video'] + side['dropped_not_physics']:,} clips set aside: "
        f"{side['dropped_not_physics']:,} whose prompts describe no motion and whose flagged moments "
        f"are not actions, "
        f"{side['dropped_no_video']} whose video could not be retrieved",
        f"{side['audio_only_flaws']} flaw records set aside because the complaint is about the "
        f"sound only, with nothing to see in the picture; no clip is dropped at this step",
        f"every flaw kept; {side['continuity_clips']} of these clips ({side['continuity_flaws']} flaws) "
        f"also form a continuity-focused group carried across experiments",
    ]
    for i, txt in enumerate(drops):
        y_from = rows[i] - 0.012
        y_to = rows[i + 1] + BAR_H + 0.012
        fig.add_artist(plt.Line2D([COL1_X + 0.012, COL1_X + 0.012], [y_from, y_to],
                                  transform=fig.transFigure, color=ADOBE_RED, lw=0.8))
        fig.add_artist(plt.Line2D([COL1_X + 0.012], [y_to], transform=fig.transFigure,
                                  marker="v", ms=3.6, color=ADOBE_RED, lw=0))
        note_x = COL1_X + 0.030
        lines = _wrap(fig, txt, (1 - 0.030) - note_x, _serif(7.4))
        y_mid = (y_from + y_to) / 2 + 0.014 * (len(lines) - 1)
        for j, line in enumerate(lines):
            fig.text(note_x, y_mid - j * 0.028, line, ha="left", va="center",
                     fontproperties=_serif(7.4), color=GRAY)

    fig.add_artist(plt.Line2D([L, 1 - 0.030], [0.088, 0.088], transform=fig.transFigure,
                              color=FILL_MID, lw=0.7))
    foot = (f"A further {side['train_clips']} annotated clips ({side['train_flaws']:,} flaws) are held back for "
            f"training and share no clip with the frozen evaluation set or its routine core. "
            f"Bar lengths are to scale within each column.")
    for j, line in enumerate(_wrap(fig, foot, (1 - 0.030) - L, _serif(7.6))):
        fig.text(L, 0.048 - j * 0.036, line, fontproperties=_serif(7.6), color=GRAY, va="baseline")
    save(fig, "fig_benchmark_composition")


def _text_w(fig, s: str, fp) -> float:
    """Width of `s` in figure coordinates, measured with the real renderer."""
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    t = fig.text(0, 0, s, fontproperties=fp)
    bb = t.get_window_extent(renderer=rend)
    t.remove()
    return inv.transform((bb.width, 0))[0] - inv.transform((0, 0))[0]


def _wrap(fig, text: str, max_w: float, fp) -> list[str]:
    """Greedy word wrap to a measured figure-space width, so nothing can clip."""
    out, cur = [], ""
    for w in text.split():
        cand = f"{cur} {w}".strip()
        if cur and _text_w(fig, cand, fp) > max_w:
            out.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        out.append(cur)
    return out


# --------------------------------------------------------------------------- #
# (b) anatomy of one human flaw record
# --------------------------------------------------------------------------- #
def _flow_text(fig, x0, y0, width, words, fp_normal, fp_hi, line_h, color, color_hi, hi_flags):
    """Lay out words left-to-right, wrapping at `width`, highlighting some.

    Returns the last baseline and, per line, the extent of the highlighted run
    so it can be underlined as one continuous rule. Widths come from the real
    renderer, so text can neither overlap nor run off the figure.
    """
    space_n = _text_w(fig, "n", fp_normal) * 0.44
    x, y = x0, y0
    runs: dict[float, list[float]] = {}
    for word, hi in zip(words, hi_flags):
        fp = fp_hi if hi else fp_normal
        w = _text_w(fig, word, fp)
        if x > x0 and x + w > x0 + width:
            x, y = x0, y - line_h
        fig.text(x, y, word, fontproperties=fp, color=color_hi if hi else color, va="baseline")
        if hi:
            lo, hiX = runs.get(y, (x, x + w))
            runs[y] = (min(lo, x), max(hiX, x + w))
        x += w + space_n
    return y, [(lo, yy, hiX - lo) for yy, (lo, hiX) in runs.items()]


def fig_anatomy(rec: dict, core_stats: dict) -> None:
    FIG_W, FIG_H = 6.5, 4.15
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ASPECT = FIG_W / FIG_H  # to keep circles round in figure coordinates

    L, R = 0.042, 0.968
    W = R - L
    LH = 0.050  # one text line, in figure units

    def field(y, name):
        fig.text(L, y, name.upper(), fontproperties=_semibold(7.6), color=GRAY,
                 va="baseline")

    fig.text(L, 0.955, "What one human flaw record contains",
             fontproperties=_semibold(10.5), color=INK, va="baseline")
    fig.text(R, 0.955, f"one of the {core_stats['n_flaws']} records in the evaluation core",
             fontproperties=_serif(7.4), color=GRAY, va="baseline", ha="right")
    fig.add_artist(plt.Line2D([L, R], [0.933, 0.933], color=INK, lw=0.8,
                              transform=fig.transFigure))

    # ---- the prompt, with the violated fragment picked out -----------------
    field(0.888, "the prompt the video was asked to satisfy")
    prompt, frag = rec["prompt"], rec["span_text"]
    i = prompt.index(frag)
    j = i + len(frag)
    # keep punctuation that immediately follows the fragment attached to its
    # last word, so the sentence still reads correctly
    tail = ""
    while j < len(prompt) and prompt[j] in ".,;:!?":
        tail += prompt[j]
        j += 1
    words, flags = [], []
    for chunk, is_hi in ((prompt[:i], False), (frag, True), (prompt[j:], False)):
        chunk_words = chunk.split()
        for k, w in enumerate(chunk_words):
            words.append(w)
            flags.append(is_hi)
        if is_hi and tail and words:
            words[-1] = words[-1] + tail
    last_y, hi_spans = _flow_text(
        fig, L, 0.888 - LH, W, words,
        _serif(9.2), _serif_bold(9.2), LH, INK, ADOBE_RED, flags,
    )
    for (hx, hy, hw) in hi_spans:  # thin red rule under the violated fragment
        fig.add_artist(plt.Line2D([hx, hx + hw], [hy - 0.017, hy - 0.017],
                                  color=ADOBE_RED, lw=1.0, transform=fig.transFigure))
    fig.text(R, last_y - 0.046, "the fragment the video fails to satisfy",
             fontproperties=_semibold(7.6), color=ADOBE_RED, va="baseline", ha="right")

    # ---- the clip, its timeline, and the flagged window -------------------
    t_top = last_y - 0.100
    field(t_top, "where the flaw occurs")

    dur, fps = rec["duration_s"], rec["fps"]
    thumb_frames = rec["key_frames"]
    frames = read_frames(rec["video"], thumb_frames)

    STRIP_Y, STRIP_H = t_top - 0.268, 0.212
    if frames:
        # the three frames the annotator boxed, laid out above the moment each
        # one occupies on the timeline, with the annotator's own box drawn on
        n = len(thumb_frames)
        tw = 0.183
        gap = (W - n * tw) / max(n - 1, 1)
        for k, fi in enumerate(thumb_frames):
            x0 = L + k * (tw + gap)
            axi = fig.add_axes([x0, STRIP_Y, tw, STRIP_H])
            axi.imshow(frames[fi], aspect="auto")
            box = rec["box_by_frame"].get(fi)
            if box:
                h, w = frames[fi].shape[:2]
                axi.add_patch(Rectangle((box[0] * w, box[1] * h),
                                        (box[2] - box[0]) * w, (box[3] - box[1]) * h,
                                        fill=False, edgecolor=ADOBE_RED, lw=1.1))
            axi.set_xticks([])
            axi.set_yticks([])
            for sp in axi.spines.values():
                sp.set_color(GRAY)
                sp.set_linewidth(0.6)
            axi.set_title(f"the video at {fi / fps:.2f} s", fontproperties=_semibold(7.8),
                          color=GRAY, pad=3.0)
            # a thin leader tying this frame to the instant it occupies below
            xt = L + (fi / fps) / dur * W
            fig.add_artist(plt.Line2D([x0 + tw / 2, xt], [STRIP_Y - 0.006, STRIP_Y - 0.052],
                                      transform=fig.transFigure, color="#DCDCDC", lw=0.6,
                                      zorder=0))

    ax = fig.add_axes([L, STRIP_Y - 0.122, W, 0.062])
    ax.add_patch(Rectangle((0, 0), dur, 1, facecolor=FILL_LIGHT, edgecolor=GRAY, lw=0.6, zorder=1))
    ax.add_patch(Rectangle((rec["start_s"], 0), rec["span_s"], 1,
                           facecolor=ADOBE_RED, alpha=0.22, edgecolor="none", zorder=2))
    for xx in (rec["start_s"], rec["end_s"]):
        ax.plot([xx, xx], [0, 1], color=ADOBE_RED, lw=1.2, zorder=4)
    for ks in rec["key_frame_s"]:
        ax.plot([ks], [0.5], marker="o", ms=3.6, mfc="white", mec=ADOBE_RED, mew=1.0, zorder=5)
    ax.set_xlim(0, dur)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ticks = [t for t in range(0, int(dur) + 1, 2)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t} s" for t in ticks], fontproperties=_serif(8.0))
    ax.tick_params(axis="x", length=2.2, pad=2, colors=GRAY)
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(False)
    article = "an" if f"{dur:.1f}".startswith("8") else "a"
    ax.annotate(
        f"flagged {rec['start_s']:.2f}–{rec['end_s']:.2f} s  —  {rec['span_s']:.2f} s of "
        f"{article} {dur:.1f} s clip",
        xy=((rec["start_s"] + rec["end_s"]) / 2, 1.16), xycoords="data", ha="center", va="bottom",
        fontproperties=_semibold(8.2), color=ADOBE_RED, annotation_clip=False,
        bbox=dict(boxstyle="square,pad=0.24", facecolor="white", edgecolor="none"),
    )
    ax.annotate(
        "the rest of the clip is not flagged",
        xy=(dur, 0.5), xycoords="data", xytext=(-4, 0), textcoords="offset points",
        ha="right", va="center", fontproperties=_serif(7.5), color=GRAY, annotation_clip=False,
    )

    # ---- severity + rationale --------------------------------------------
    s_top = STRIP_Y - 0.212
    field(s_top, "how bad it is, and why")
    dot_y = s_top - 0.042
    fig.text(L, dot_y, "severity", fontproperties=_serif(8.4), color=GRAY, va="center")
    x = L + 0.082
    for k in range(1, 6):
        filled = k <= rec["severity"]
        fig.add_artist(Ellipse((x, dot_y), 0.0135, 0.0135 * ASPECT, transform=fig.transFigure,
                               facecolor=INK if filled else "white",
                               edgecolor=INK if filled else "#C8C8C8", lw=0.8))
        x += 0.0215
    fig.text(x + 0.005, dot_y, f"{rec['severity']} of 5", fontproperties=_semibold(8.4),
             color=INK, va="center")
    fig.text(x + 0.082, dot_y, f"annotator confidence {rec['confidence']} of 5",
             fontproperties=_serif(8.4), color=GRAY, va="center")

    fig.text(L, dot_y - 0.040, "the annotator's own reasoning, as written",
             fontproperties=_semibold(7.2), color=GRAY, va="baseline")
    rationale = '\u201c' + rec["reasoning"].rstrip() + '\u201d'
    if len(rationale) > 260:
        rationale = rationale[:257].rsplit(" ", 1)[0] + "…"
    words = rationale.split()
    _flow_text(fig, L, dot_y - 0.076, W, words,
               _serif(8.4, italic=True), _serif(8.4, italic=True), 0.042,
               GRAY, GRAY, [False] * len(words))

    save(fig, "fig_benchmark_anatomy")


# --------------------------------------------------------------------------- #
# (c) distributions over the routine core
# --------------------------------------------------------------------------- #
def fig_distributions(cs: dict) -> None:
    fig = plt.figure(figsize=(6.5, 2.95))
    ax1 = fig.add_axes([0.082, 0.325, 0.352, 0.500])
    ax2 = fig.add_axes([0.600, 0.325, 0.352, 0.500])

    # -- left: how many flaws a clip carries -------------------------------
    hist = cs["per_clip_hist"]
    xs = list(range(1, cs["per_clip_max"] + 1))
    ys = [hist.get(k, 0) for k in xs]
    ax1.bar(xs, ys, width=0.68, color=FILL_MID, edgecolor=GRAY, linewidth=0.6, zorder=2)
    for x, y in zip(xs, ys):
        if not y:
            continue
        # nudge a count off the average rule if the rule would run through it
        dx = -0.16 if abs(x - cs["per_clip_mean"]) < 0.10 else 0.0
        ax1.text(x + dx, y + max(ys) * 0.032, str(y), ha="center", va="bottom",
                 fontproperties=_semibold(8.0), color=GRAY)
    ax1.set_xlabel("flaws marked in one clip", fontproperties=_semibold(8.6), labelpad=4)
    ax1.set_ylabel("clips", fontproperties=_semibold(8.6), labelpad=3)
    ax1.set_xticks(xs)
    ax1.set_xlim(0.4, cs["per_clip_max"] + 0.6)
    ax1.set_ylim(0, max(ys) * 1.18)
    ax1.set_title("Most clips carry one or two flaws",
                  fontproperties=_semibold(9.2), color=INK, loc="left", pad=8)
    # the average, marked where the panel is empty so no label can collide
    _marker_line(ax1, cs["per_clip_mean"], max(ys) * 1.18,
                 f"average {cs['per_clip_mean']:.2f} per clip", side="right")

    # -- right: how much of the clip a flaw occupies -----------------------
    fr = cs["span_frac"]
    bins = [i / 10 for i in range(11)]
    counts, _ = _histcounts(fr, bins)
    ax2.bar([b + 0.05 for b in bins[:-1]], counts, width=0.092, color=FILL_MID,
            edgecolor=GRAY, linewidth=0.6, zorder=2)
    ymax = max(counts)
    ax2.set_xlabel("share of the clip the flagged window covers",
                   fontproperties=_semibold(8.6), labelpad=4)
    ax2.set_ylabel("flaws", fontproperties=_semibold(8.6), labelpad=3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, ymax * 1.18)
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_xticklabels(["none", "25%", "50%", "75%", "all"])
    ax2.set_title("Flagged windows are either brief or clip-long",
                  fontproperties=_semibold(9.2), color=INK, loc="left", pad=8)
    _marker_line(ax2, cs["span_frac_median"], ymax * 1.18,
                 f"median {cs['span_frac_median'] * 100:.0f}%", side="left")

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(length=2.4, pad=2.5)

    note = (f"{cs['n_clips']} clips of the routine evaluation core, {cs['n_flaws']} human-marked flaws. "
            f"The right-hand panel uses the {cs['n_timed']} flaws that carry a start and an end frame; the "
            f"remaining {cs['n_untimed']} were marked without one. Of those {cs['n_timed']}, "
            f"{cs['frac_le_25']} fit inside a quarter of their clip and {cs['frac_ge_90']} run almost its "
            f"whole length; the median flagged window lasts {cs['span_s_median']:.2f} s.")
    for j, line in enumerate(_wrap(fig, note, 0.870, _serif(7.6))):
        fig.text(0.082, 0.118 - j * 0.052, line, fontproperties=_serif(7.6), color=GRAY,
                 va="baseline")
    save(fig, "fig_benchmark_distributions")



def _marker_line(ax, x, y_top, label, side="right"):
    """A thin red rule at `x` with its label tucked into the empty upper corner."""
    ax.plot([x, x], [0, y_top * 0.995], color=ADOBE_RED, lw=0.9, zorder=4, clip_on=False)
    dx = 0.012 * (ax.get_xlim()[1] - ax.get_xlim()[0])
    ax.text(x + (dx if side == "right" else -dx), y_top * 0.96, label,
            ha="left" if side == "right" else "right", va="top",
            fontproperties=_semibold(8.0), color=ADOBE_RED, zorder=5)


def _histcounts(vals, bins):
    counts = [0] * (len(bins) - 1)
    for v in vals:
        for i in range(len(bins) - 1):
            hi_incl = i == len(bins) - 2
            if bins[i] <= v < bins[i + 1] or (hi_incl and v == bins[i + 1]):
                counts[i] += 1
                break
    return counts, bins


# --------------------------------------------------------------------------- #
def save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=200)
    plt.close(fig)
    print(f"  wrote {OUT / (stem + '.pdf')}")
    print(f"  wrote {OUT / (stem + '.png')}")


def main() -> None:
    fonts = register_source_fonts()
    print("=" * 78)
    print("FONTS registered (report faces, not matplotlib defaults):",
          len(fonts), "files;", Path(next(iter(fonts.values()))).parent)
    D = load_all()
    print("SOURCES")
    for k, v in SOURCES.items():
        print(f"  {k:11s} {v}")

    stages = compute_funnel(D)
    side = compute_side_counts(D)
    print("\n--- (a) COMPOSITION  [clips from record counts; flaws summed per clip]")
    for s in stages:
        print(f"  {s['label']:28s} clips={s['clips']:5d}  flaws={s['flaws']:5d}")
    print("  drop accounting (pilot manifest 'stats' + flaw scope tally):")
    print(f"    no retrievable video      {side['dropped_no_video']}")
    print(f"    not physics-relevant      {side['dropped_not_physics']}")
    print(f"    audible-only flaws        {side['audio_only_flaws']}   scope tally={side['scope_hist']}")
    print(f"    1500 - {side['dropped_no_video']} - {side['dropped_not_physics']} = "
          f"{1500 - side['dropped_no_video'] - side['dropped_not_physics']}  (pool clips)")
    print(f"    {stages[1]['flaws']} - {side['audio_only_flaws']} = {stages[2]['flaws']}  (frozen flaws)")
    print(f"    continuity stratum        {side['continuity_clips']} clips / {side['continuity_flaws']} flaws")
    print(f"    training pool             {side['train_clips']} clips / {side['train_flaws']} flaws; "
          f"overlap with frozen eval={side['train_overlap_eval']}, with core={side['train_overlap_core']}")
    print(f"    core is subset of frozen eval: {side['core_subset_of_eval']}")

    cs = compute_core_stats(D)
    print("\n--- (c) CORE STATISTICS  [gold_core_v1 'flaws' joined to raw start/end_frame_index]")
    print(f"  clips={cs['n_clips']}  flaws={cs['n_flaws']}  exact joins verified={cs['joined']}")
    print(f"  flaws per clip: {cs['per_clip_hist']}")
    print(f"    mean={cs['per_clip_mean']:.4f} median={cs['per_clip_median']} max={cs['per_clip_max']}")
    print(f"  clip length s: min={cs['dur_min']:.3f} median={cs['dur_median']:.3f} max={cs['dur_max']:.3f}")
    print(f"  flaws with a time span={cs['n_timed']}  without={cs['n_untimed']}")
    print(f"  span seconds: min={cs['span_s_min']:.3f} median={cs['span_s_median']:.3f} "
          f"mean={cs['span_s_mean']:.3f} max={cs['span_s_max']:.3f}")
    print(f"  span share of clip: min={cs['span_frac_min']:.4f} median={cs['span_frac_median']:.4f} "
          f"mean={cs['span_frac_mean']:.4f} max={cs['span_frac_max']:.4f}")
    print(f"    <=25% of clip: {cs['frac_le_25']}   <=50%: {cs['frac_le_50']}   >=90%: {cs['frac_ge_90']}")
    print(f"  machine-derived category tally: {cs['cat_hist']}")
    print(f"  severity tally: {cs['sev_hist']}")

    rec = pick_anatomy(D)
    print("\n--- (b) ANATOMY RECORD  [verbatim from the files]")
    for k in ("item_id", "severity", "confidence", "fps", "duration_s", "start_frame",
              "end_frame", "start_s", "end_s", "span_s", "n_boxed_keyframes",
              "key_frames", "n_flaws_in_clip"):
        print(f"  {k:18s} {rec[k]}")
    print(f"  span_text          {rec['span_text']!r}")
    print(f"  prompt             {rec['prompt']!r}")
    print(f"  reasoning          {rec['reasoning']!r}")

    print("\n--- RENDER")
    fig_composition(stages, side)
    fig_anatomy(rec, cs)
    fig_distributions(cs)
    print("=" * 78)


if __name__ == "__main__":
    main()
