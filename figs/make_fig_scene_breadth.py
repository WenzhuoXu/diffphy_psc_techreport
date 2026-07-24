#!/usr/bin/env python3
"""Build the two scene-breadth figures for the technical report.

The generation section otherwise shows only two rigid balls. These two figures
show the rest of the scene set -- cloth, cable and chain, granular piles,
buoyancy, air drag, toppling chains, a chaotic linkage -- and they are made only
of REAL frames decoded from the real clips.

    fig_scene_breadth        -- a contact sheet: one tile per scene kind, twelve
                                kinds, each tile a frame of the actual generated
                                video for that kind.
    fig_scene_control_pairs  -- three of those kinds shown the way the report's
                                other generation filmstrips show a case: the
                                simulated depth control on top, the generated
                                video below, at matched timestamps.

Figure B reuses the block template of make_fig_generation.py (imported, not
copied) so the three figures read as one family.

WHAT IS AND IS NOT BEING CLAIMED. Everything here comes out of one rigid-body
physics engine, so the deformables are that engine's PARTICLE-AND-LINK
primitives: cloth is a sheet of particles joined by springs, a rope or chain is a
short row of linked segments, sand and marbles are many small solid bodies, and
floating is a buoyancy force on a solid body in a still fluid volume. None of it
is a fluid solve, none of it is smoke, and none of it is a finite-element soft
body. Every label in both figures is written to say exactly that much and no
more, and the tile captions are read out of the source builder file rather than
written by hand -- see PRIMITIVE_LINES.

Which frame each tile shows is COMPUTED, not chosen by hand: for every clip the
frame-to-frame change is measured over the whole clip, accumulated, and the tile
is the frame at which half of the clip's total change has happened. That lands
mid-drape / mid-swing / mid-pour rather than on frame 0, and it is the same rule
for all twelve clips. The three timestamps in figure B are picked the same way,
at a quarter, a half and three quarters of the accumulated change.

Everything printed to stdout is measured from the files: the resolved font files,
each clip's frame count / rate / size, the eye-check verdict and its verbatim
reason for every tile, the chosen frame index, how bright and how varied that
frame is (so a blank or near-blank tile could not pass unnoticed), and for figure
B a check that each control render really is a depth render (grey, not colour)
and lines up with its generated clip frame for frame.

Run:  python3 /Users/wenzhuox/diffphy_psc/techreport/figs/make_fig_scene_breadth.py

Idempotent: re-running overwrites the same four output files.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path("/Users/wenzhuox/diffphy_psc/techreport/figs")
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# The report's visual system and the shared filmstrip block template live in the
# generation figure script; both are imported so there is exactly one copy.
from make_fig_generation import (  # noqa: E402
    CONTENT_W, FIG_W, GRAY, HAIRLINE, INK, N_COLS, PAD_IN,
    Layout, apply_style, build_figure, probe, read_all, save, semibold,
)

# --------------------------------------------------------------------------- #
# paths (absolute; sources are read-only)
# --------------------------------------------------------------------------- #
OUT_DIR = HERE
ASSETS = HERE / "assets/scene_breadth"
GEN = ASSETS / "gen"            # generated videos, one per scene
CTRL = ASSETS / "ctrl"          # the simulated depth control each one was made from

# Where the two asset folders came from, so they can be refilled if lost. Only
# the twelve clips these figures draw from are kept here; the whole set (167
# generated clips and 167 controls) stays in the bucket it was produced into.
#   generated:  s3://jiuxiang/interns/wenzhuo/exp029sl/w22sl_shard*/<name>.mp4
#   controls:   s3://jiuxiang/interns/wenzhuo/exp029sl/controls/<name>_depth.mp4
# The generated clips are 848x480, 81 frames at 16 fps; each control is the
# MuJoCo depth render of the same simulation at the same size and length, so the
# two line up frame for frame (checked at run time, not assumed).

SCENE_LIB = Path("/Users/wenzhuox/diffphy_exp028/artifacts/runs/exp028/scene_lib")
EYECHECK = SCENE_LIB / "eyecheck.jsonl"
METRICS = SCENE_LIB / "metrics_all.jsonl"
BUILDERS = Path("/Users/wenzhuox/diffphy_exp028/scripts/exp028/gen_scene_library.py")

# --------------------------------------------------------------------------- #
# what the physics primitive actually is, quoted from the builder file
# --------------------------------------------------------------------------- #
# Each entry is (scene kind -> the name of the function in the builder file that
# builds it, and a substring that must appear VERBATIM inside THAT function).
# Scoping the search to the one function is the point: a bare 'type="sphere"'
# occurs in dozens of unrelated builders and would prove nothing, whereas
# 'r=0.05' inside sand_pile is that scene's own grain radius. The check below
# fails loudly if the function is missing or the substring is not in it, so no
# tile can be captioned with a primitive that is not in the source.
PRIMITIVE_LINES = {
    "cloth_drape":   ("cloth_drape",     'kind="cloth"'),
    "cloth_fall":    ("cloth_fall",      'kind="cloth"'),
    "flag":          ("flag_wind",       'kind="cloth"'),
    "rope_swing":    ("rope_swing",      'kind="cable"'),
    "hanging_chain": ("hanging_chain",   'kind="cable"'),
    "sand_pile":     ("sand_pile",       'type="sphere", r=0.05'),
    "hourglass":     ("hourglass",       'type="sphere", r=0.035'),
    "ball_pit":      ("ball_pit_splash", 'type="sphere", r=0.1'),
    "boat_float":    ("boat_float",      "density=1000"),
    "avalanche":     ("avalanche",       'type="sphere", r=0.075'),
    "dominoes":      ("dominoes",        'type="box", sx=0.012'),
    "double_pend":   ("double_pendulum", "chain=[L1,L2]"),
}


def builder_body(src: str, func: str) -> str:
    """The text of one `def <func>(...)` block in the builder file."""
    lines = src.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith(f"def {func}(")), None)
    if start is None:
        raise SystemExit(f"there is no 'def {func}(' in {BUILDERS}")
    end = next((j for j in range(start + 1, len(lines))
                if lines[j].startswith("def ") or lines[j].startswith("# =====")),
               len(lines))
    return "\n".join(lines[start:end])

# --------------------------------------------------------------------------- #
# the twelve tiles of figure A
# --------------------------------------------------------------------------- #
# `clip` is the generated video's stem; `eye` is the name of the eye-check row
# relied on (for the camera-moving variants the reviewed name carries the `_cm`
# suffix, so it is spelled out rather than guessed); `label` is a plain-English
# scene name with no file name, no code identifier and no claim of fluid or
# finite-element simulation; `sub` names the primitive in plain words.
# Two things a tile must NOT do, and both were caught by tracking the bodies in
# the matching control render frame by frame before choosing the tile:
#  - claim a phenomenon the clip does not show. An "a big light body falls slower
#    than a small heavy one" tile was cut: in its control the two bodies descend
#    together, both centres going from y=66 to y=221 between frame 0 and frame 12,
#    within about 3 px of each other. That is not air drag telling them apart, so
#    the label would have captioned physics that is not on screen. A separate
#    "some bodies float, some sink" tile was cut for the same reason: nothing in
#    it sinks -- the two dense bodies drop from y=128 and stop at y=191, the same
#    height the two light bodies sit at, and after frame 20 nothing moves again.
#  - be shown at a frame where the phenomenon has not happened yet or is already
#    over. That is what the change rule is for.
# Buoyancy is instead shown by the boat, where the loading is what is visible: in
# its control the two cargo blocks fall from y=23 to y=158 over the first twelve
# frames and merge into the hull, and the hull's own centre moves down from y=226
# to y=230 and stays there -- the boat takes the load and keeps floating.
TILES = [
    dict(clip="cloth_drape_016_cm", eye="cloth_drape_016_cm", kind="cloth_drape",
         label="Cloth draped over a cylinder",  sub="particle sheet"),
    dict(clip="cloth_fall_000",     eye="cloth_fall_000",     kind="cloth_fall",
         label="Cloth falling, crumpling",      sub="particle sheet"),
    dict(clip="flag_001_cm",        eye="flag_001_cm",        kind="flag",
         label="Flag rippling in wind",         sub="pinned particle sheet"),
    dict(clip="rope_swing_009",     eye="rope_swing_009",     kind="rope_swing",
         label="Rope swinging, whipping",       sub="linked segments"),
    dict(clip="hanging_chain_002",  eye="hanging_chain_002",  kind="hanging_chain",
         label="Chain hanging, swinging",       sub="linked segments"),
    # `count` names the scene parameter that the number in `sub` has to equal, and
    # `count_plus` any bodies on top of it (the pit's own dropped ball). The check
    # below reads the number back out of the caption and compares, so a count in
    # print can only ever be the count in the scene's own parameters.
    dict(clip="sand_pile_002",      eye="sand_pile_002",      kind="sand_pile",
         label="Grains pouring in a heap",      sub="34 small bodies",
         count="n"),
    dict(clip="hourglass_001",      eye="hourglass_001",      kind="hourglass",
         label="Grains draining a funnel",      sub="30 small bodies",
         count="n"),
    dict(clip="ball_pit_003",       eye="ball_pit_003",       kind="ball_pit",
         label="Ball into a pit of balls",      sub="9 bodies in contact",
         count="n_ring", count_plus=1),
    dict(clip="boat_float_001",     eye="boat_float_001",     kind="boat_float",
         label="Loaded boat stays afloat",      sub="buoyancy force"),
    dict(clip="avalanche_005",      eye="avalanche_005",      kind="avalanche",
         label="Cluster down a slope",          sub="6 bodies on a ramp",
         count="n"),
    dict(clip="dominoes_023",       eye="dominoes_023",       kind="dominoes",
         label="Dominoes toppling in turn",     sub="20 rigid slabs",
         count="n"),
    dict(clip="double_pend_015",    eye="double_pend_015",    kind="double_pend",
         label="Linkage swinging chaotically",  sub="two jointed arms"),
]

# --------------------------------------------------------------------------- #
# the three cases of figure B  (control above, generated below)
# --------------------------------------------------------------------------- #
# These three are the ones whose control render still SHOWS its object at every
# column. The control is a depth picture -- near is bright, far is dark -- so an
# object that comes to rest on the floor ends up the same grey as the floor right
# behind it and simply disappears from the control, even though the simulation is
# perfectly fine. A pair like that would put a blank panel on the page above a
# perfectly good generated frame, so how much of each control column is
# distinguishable from its background is MEASURED (control_object_fraction) and
# checked against PAIR_MIN_OBJECT below. A cloth-onto-the-floor and a
# grains-onto-the-floor case were both cut by that check: their controls fall to
# 0.00% and 0.61% of the frame in the last columns, against 4.9% and 8.2% for the
# cloth-over-a-cylinder and funnel cases kept here.
PAIRS = [
    dict(clip="cloth_drape_016_cm", eye="cloth_drape_016_cm", kind="cloth_drape",
         title="Cloth falling and draping over a cylinder"),
    dict(clip="rope_swing_009",     eye="rope_swing_009",     kind="rope_swing",
         title="Rope swinging from a fixed anchor"),
    dict(clip="hourglass_001",      eye="hourglass_001",      kind="hourglass",
         title="Grains draining through a funnel"),
]

# The smallest share of a control frame that may be object rather than background.
# 1% of an 848x480 frame is about 4,000 pixels -- comfortably enough to read a
# shape at the printed panel size, and far above the 0.0-0.6% that a vanished
# object leaves.
PAIR_MIN_OBJECT = 0.01

# --------------------------------------------------------------------------- #
# figure A geometry -- same family as the generation filmstrips
# --------------------------------------------------------------------------- #
A_COLS = 4
A_COL_GAP = 0.075        # same column gap as the filmstrips
A_CAP_GAP = 0.075        # frame -> its caption
A_CAP_H = 0.235          # two caption lines
A_ROW_GAP = 0.155        # caption -> next row of frames
A_HEAD_BASE = 0.105      # heading baseline, as in the filmstrips
A_HEAD_RULE = 0.160      # hairline under the heading
A_SUB_BASE = 0.320       # the one-line note under the rule (baseline)
A_HEAD_H = 0.430         # heading band; = A_SUB_BASE + clearance, so the note's
                         # descenders cannot touch the first row of frames
A_LABEL_FS = 8.0
A_SUB_FS = 8.0
A_HEAD_FS = 8.8


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #
def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def change_profile(frames: list[np.ndarray]) -> np.ndarray:
    """How much the picture changes from each frame to the next, accumulated.

    Lightly blurred greyscale, mean absolute difference between neighbouring
    frames, then a running total. The blur is there so film grain and codec
    noise do not swamp the real motion.
    """
    grey = [cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (0, 0), 1.2).astype(np.float32)
            for f in frames]
    step = np.array([0.0] + [float(np.abs(grey[i] - grey[i - 1]).mean())
                             for i in range(1, len(grey))])
    return np.cumsum(step)


def frame_at_change(cum: np.ndarray, fraction: float) -> int:
    """The frame by which `fraction` of the clip's whole change has happened."""
    return int(np.searchsorted(cum, fraction * cum[-1]))


def frame_stats(frame: np.ndarray) -> tuple[float, float, float]:
    """Brightness, spread of brightness, and how much of the frame has an edge.

    A blank or near-blank tile would show up as a tiny spread and almost no
    edges, so these three numbers are printed for every tile as the check that
    each one really carries a picture.
    """
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(g, 60, 160)
    return float(g.mean()), float(g.std()), float((edges > 0).mean())


def control_object_fraction(frame: np.ndarray) -> float:
    """What share of a depth control frame is object rather than background.

    The background of these depth renders is a smooth top-to-bottom ramp -- every
    row of it is one grey -- so a pixel that differs from its own row's median is
    an object pixel. That is what makes it possible to say whether a control panel
    still shows anything, without needing to know what the object is.
    """
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    row_bg = np.median(g, axis=1, keepdims=True)
    return float((np.abs(g - row_bg) > 6).mean())


def is_greyscale(frame: np.ndarray, tol: int = 3) -> bool:
    """True if the frame carries no colour -- i.e. it is a depth/mask render."""
    b, g, r = frame[:, :, 0].astype(np.int16), frame[:, :, 1].astype(np.int16), frame[:, :, 2].astype(np.int16)
    return bool(np.abs(b - g).max() <= tol and np.abs(g - r).max() <= tol)


# --------------------------------------------------------------------------- #
# figure A
# --------------------------------------------------------------------------- #
def a_col_w() -> float:
    return (CONTENT_W - A_COL_GAP * (A_COLS - 1)) / A_COLS


def text_width_in(fig, artist) -> float:
    """How wide a drawn string actually is, in inches, as rendered.

    Measured off the real renderer rather than estimated, so a caption that would
    run past its column is a hard error instead of something to notice later in
    the PNG.
    """
    fig.canvas.draw()
    bb = artist.get_window_extent(fig.canvas.get_renderer())
    return bb.width / fig.dpi


def build_contact_sheet(tiles: list[dict], aspect: float, n_fam: int,
                        stem: str) -> list[Path]:
    """The contact sheet: one real frame per scene kind, four to a row."""
    import matplotlib.pyplot as plt

    cw = a_col_w()
    ch = cw / aspect
    n_rows = (len(tiles) + A_COLS - 1) // A_COLS
    content_h = A_HEAD_H + n_rows * (ch + A_CAP_GAP + A_CAP_H) + (n_rows - 1) * A_ROW_GAP
    fig_h = content_h + 2 * PAD_IN
    fig = plt.figure(figsize=(FIG_W, fig_h))
    L = Layout(fig, FIG_W, fig_h)

    L.text(0.0, A_HEAD_BASE, "Beyond a single ball", ha="left", va="baseline",
           fontproperties=semibold(A_HEAD_FS), color=INK)
    L.text(CONTENT_W, A_HEAD_BASE,
           f"{len(tiles)} of {n_fam} scene kinds", ha="right", va="baseline",
           fontsize=A_LABEL_FS, color=GRAY)
    L.hrule(0.0, CONTENT_W, A_HEAD_RULE, GRAY)
    # The accent is deliberately NOT used in this figure. In the filmstrips it
    # means one specific thing -- this row is the model's output -- and there is
    # no such distinction to mark on a sheet where every tile is an output. A red
    # rule here would be decoration, so there isn't one.
    L.text(0.0, A_SUB_BASE,
           "One frame of the generated video per scene, at the point half the clip's "
           "motion has happened.",
           ha="left", va="baseline", fontsize=A_SUB_FS, color=GRAY)

    captions = []
    for k, t in enumerate(tiles):
        r, c = divmod(k, A_COLS)
        x = c * (cw + A_COL_GAP)
        y = A_HEAD_H + r * (ch + A_CAP_GAP + A_CAP_H + A_ROW_GAP)
        ax = fig.add_axes(L.rect(x, y, cw, ch))
        ax.imshow(cv2.cvtColor(t["frame"], cv2.COLOR_BGR2RGB), interpolation="antialiased")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(GRAY); s.set_linewidth(HAIRLINE)
        captions.append((t["label"],
                         L.text(x, y + ch + A_CAP_GAP + 0.075, t["label"], ha="left",
                                va="baseline", fontsize=A_LABEL_FS, color=INK)))
        captions.append((t["sub"],
                         L.text(x, y + ch + A_CAP_GAP + 0.200, t["sub"], ha="left",
                                va="baseline", fontsize=A_SUB_FS, color=GRAY)))

    # Every caption has to fit inside the column it labels, measured as rendered.
    print(f"  [{stem}] caption widths, measured as drawn (column is {cw:.3f} in):")
    worst = 0.0
    for s, art in captions:
        w = text_width_in(fig, art)
        worst = max(worst, w)
        flag = "  <-- OVERFLOWS" if w > cw else ""
        print(f"      {w:5.3f} in  \"{s}\"{flag}")
    if worst > cw:
        raise SystemExit(
            f"a caption is {worst:.3f} in wide but its column is only {cw:.3f} in -- "
            f"shorten the label rather than shipping type that runs into the next tile"
        )
    print(f"      widest caption {worst:.3f} in of {cw:.3f} in available")

    print(f"  [{stem}] {len(tiles)} tiles, {A_COLS} per row, panel "
          f"{cw:.3f} x {ch:.3f} in, column gap {A_COL_GAP:.3f} in, "
          f"row gap {A_ROW_GAP:.3f} in")
    print(f"  [{stem}] figure {FIG_W:.2f} x {fig_h:.2f} in, content "
          f"{CONTENT_W:.2f} x {content_h:.2f} in, margin {PAD_IN:.2f} in on all four "
          f"sides, every rule {HAIRLINE:.1f} pt, smallest type {min(A_LABEL_FS, A_SUB_FS):.1f} pt")
    return save(fig, stem)


# --------------------------------------------------------------------------- #
def main() -> int:
    apply_style()

    # ---------------------------------------------------------------- sources
    print("\n" + "=" * 78)
    print("SCENE SET  (measured from the scene index and the review pass)")
    print("=" * 78)
    metrics = {m["name"]: m for m in load_jsonl(METRICS)}
    eyecheck = {e["name"]: e for e in load_jsonl(EYECHECK)}
    n_fam = len({m["family"] for m in metrics.values()})
    n_good = sum(1 for e in eyecheck.values() if e["good"])
    print(f"  {METRICS}")
    print(f"    {len(metrics)} scenes indexed, {n_fam} distinct scene kinds")
    print(f"  {EYECHECK}")
    print(f"    {len(eyecheck)} scenes reviewed by eye, {n_good} judged good")
    # Only the clips these two figures actually draw from are kept next to the
    # figures; the rest of the generated set stays where it was produced.
    print(f"  {GEN}")
    print(f"    {len(list(GEN.glob('*.mp4')))} generated videos held locally "
          f"(the {len({t['clip'] for t in TILES} | {p['clip'] for p in PAIRS})} these "
          f"figures draw from)")
    print(f"  {CTRL}")
    print(f"    {len(list(CTRL.glob('*.mp4')))} matching simulated depth controls")

    # -------------------------------------------- the primitive claim, checked
    print("\n" + "=" * 78)
    print("WHAT THE PHYSICS PRIMITIVE IS  (quoted from the scene builder file)")
    print("=" * 78)
    builder_src = BUILDERS.read_text()
    print(f"  {BUILDERS}")
    for kind, (func, needle) in PRIMITIVE_LINES.items():
        body = builder_body(builder_src, func)
        hits = [ln.strip() for ln in body.splitlines() if needle in ln]
        if not hits:
            raise SystemExit(
                f"{kind}: '{needle}' is not inside def {func}() -- refusing to label a tile "
                f"with a primitive I cannot find in the source"
            )
        print(f"  {kind:16s} def {func}() contains '{needle}':")
        print(f"                   {hits[0][:150]}")

    # ---------------------------------------------------- figure A, per tile
    print("\n" + "=" * 78)
    print("FIGURE A TILES  (clip, review verdict, chosen frame, and what is in it)")
    print("=" * 78)
    aspect = None
    for t in TILES:
        gen_path = GEN / f"{t['clip']}.mp4"
        info = probe(gen_path)
        frames = read_all(gen_path)
        if len(frames) != info.n_frames:
            raise SystemExit(f"{t['clip']}: decoded {len(frames)} frames, header says "
                             f"{info.n_frames}")
        row = eyecheck.get(t["eye"])
        if row is None:
            raise SystemExit(f"{t['clip']}: no review row named {t['eye']} -- refusing to "
                             f"show a tile I cannot vouch for")
        if not row["good"]:
            raise SystemExit(f"{t['clip']}: review row {t['eye']} says good=False")
        met = metrics.get(t["eye"]) or metrics.get(t["clip"])
        if met is None or met["family"] != t["kind"]:
            raise SystemExit(f"{t['clip']}: scene kind in the index is "
                             f"{met['family'] if met else None}, not {t['kind']}")

        # Any number printed in a caption has to be the scene's own parameter.
        if t.get("count"):
            want = int(met["params"][t["count"]]) + int(t.get("count_plus", 0))
            said = int("".join(ch for ch in t["sub"] if ch.isdigit()))
            plus = f" + {t['count_plus']}" if t.get("count_plus") else ""
            if said != want:
                raise SystemExit(
                    f"{t['clip']}: the caption says {said} bodies but the scene's "
                    f"'{t['count']}' parameter is {met['params'][t['count']]}{plus} = {want}"
                )
            t["count_note"] = (f"caption says {said}; scene parameter "
                               f"'{t['count']}' = {met['params'][t['count']]}{plus} -> {want}")

        cum = change_profile(frames)
        idx = frame_at_change(cum, 0.5)
        mean, std, edge = frame_stats(frames[idx])
        if std < 8.0 or edge < 0.002:
            raise SystemExit(f"{t['clip']} frame {idx}: brightness spread {std:.1f}, edge "
                             f"fraction {edge:.4f} -- that is a blank tile, not a picture")
        t.update(frame=frames[idx], idx=idx, info=info, eye_row=row, met=met,
                 stats=(mean, std, edge))
        if aspect is None:
            aspect = info.width / info.height

        print(f"\n  {t['label']}")
        print(f"    clip            {gen_path}")
        print(f"    scene kind      {met['family']}   parameters {json.dumps(met['params'])}")
        print(f"    clip            {info.n_frames} frames  {info.fps:.2f} fps  "
              f"{info.duration_s:.3f} s  {info.width}x{info.height}")
        print(f"    review          {EYECHECK.name} row '{t['eye']}'  good={row['good']}")
        print(f"    review said     \"{row['reason']}\"")
        print(f"    caption         \"{t['label']}\" / \"{t['sub']}\"")
        if t.get("count_note"):
            print(f"    number checked  {t['count_note']}")
        print(f"    frame shown     {idx}  (t = {idx / info.fps:.3f} s), the frame by which "
              f"half of the clip's total change has happened")
        print(f"    that frame      brightness {mean:.1f}/255, spread {std:.1f}, "
              f"{100 * edge:.2f}% of pixels on an edge -- not blank")

    # ------------------------------------------------------------- figure A
    print("\n" + "=" * 78)
    print("FIGURE A")
    print("=" * 78)
    pa = build_contact_sheet(TILES, aspect, n_fam, "fig_scene_breadth")

    # ---------------------------------------------------- figure B, per case
    print("\n" + "=" * 78)
    print("FIGURE B PAIRS  (does the control render match its generated clip?)")
    print("=" * 78)
    cases = []
    for p in PAIRS:
        gen_path = GEN / f"{p['clip']}.mp4"
        ctrl_path = CTRL / f"{p['clip']}_depth.mp4"
        gi, ci = probe(gen_path), probe(ctrl_path)
        gen_frames, ctrl_frames = read_all(gen_path), read_all(ctrl_path)
        row = eyecheck[p["eye"]]

        aligned = (gi.n_frames == ci.n_frames and abs(gi.fps - ci.fps) < 1e-6
                   and gi.width == ci.width and gi.height == ci.height)
        grey = [is_greyscale(ctrl_frames[i]) for i in (0, len(ctrl_frames) // 2, -1)]
        colour = is_greyscale(gen_frames[len(gen_frames) // 2])

        print(f"\n  {p['title']}")
        print(f"    control         {ctrl_path}")
        print(f"                    {ci.n_frames} frames  {ci.fps:.2f} fps  "
              f"{ci.width}x{ci.height}")
        print(f"    generated       {gen_path}")
        print(f"                    {gi.n_frames} frames  {gi.fps:.2f} fps  "
              f"{gi.width}x{gi.height}")
        print(f"    same length, rate and size in both: {aligned}")
        if not aligned:
            raise SystemExit(f"{p['clip']}: the control and the generated clip do not line up")
        print(f"    control has no colour at the start, middle and end: {grey} "
              f"(so it is a depth render, not a colour clip)")
        if not all(grey):
            raise SystemExit(f"{p['clip']}: the control render carries colour -- that is not "
                             f"a depth control")
        print(f"    generated clip has colour: {not colour}")
        if colour:
            raise SystemExit(f"{p['clip']}: the generated clip has no colour")
        print(f"    review          {EYECHECK.name} row '{p['eye']}'  good={row['good']}")
        print(f"    review said     \"{row['reason']}\"")

        # The filmstrip template shows N_COLS frames per row, so the columns are
        # the frames at equal steps of accumulated change -- the same rule that
        # picks figure A's tile, extended to N_COLS points. Equal steps of CHANGE
        # rather than equal steps of time means no column lands in a stretch where
        # nothing is happening.
        cum = change_profile(gen_frames)
        fracs = [k / (N_COLS - 1) for k in range(N_COLS)]
        idx = sorted({frame_at_change(cum, f) for f in fracs})
        while len(idx) < N_COLS:      # a tie can collapse two columns onto one frame
            gaps = [(b - a, a, b) for a, b in zip(idx, idx[1:]) if b - a >= 2]
            if not gaps:
                break
            _, a, b = max(gaps)
            idx = sorted(set(idx) | {(a + b) // 2})
        print(f"    frames shown    {idx}  (t = {[round(i / gi.fps, 3) for i in idx]} s), "
              f"at equal steps of the clip's accumulated change "
              f"({', '.join(f'{100 * f:.0f}%' for f in fracs)})")
        for i in idx:
            gm, gs, ge = frame_stats(gen_frames[i])
            obj = control_object_fraction(ctrl_frames[i])
            flag = "  <-- CONTROL HAS GONE BLANK" if obj < PAIR_MIN_OBJECT else ""
            print(f"      frame {i:3d}  generated brightness {gm:5.1f} spread {gs:5.1f} "
                  f"edges {100 * ge:4.2f}%   control shows an object over "
                  f"{100 * obj:5.2f}% of the frame{flag}")
            if gs < 8.0:
                raise SystemExit(f"{p['clip']} frame {i}: generated frame is blank")
            if obj < PAIR_MIN_OBJECT:
                raise SystemExit(
                    f"{p['clip']} frame {i}: only {100 * obj:.2f}% of the control frame "
                    f"differs from its background -- that control panel would print blank, "
                    f"so this is not an honest pair to show"
                )
        cases.append({"title": p["title"], "ctrl": ctrl_frames, "gen": gen_frames,
                      "idx": idx, "fps": gi.fps,
                      # no height tick in this figure: these scenes have no single
                      # object whose height means anything (a sheet, a rope, a
                      # pour), so a tick would be a made-up measurement
                      "cy": None, "frame_h": gi.height})

    # ------------------------------------------------------------- figure B
    print("\n" + "=" * 78)
    print("FIGURE B  (the filmstrip block template, imported from the other figure script)")
    print("=" * 78)
    pb = build_figure(cases, aspect, "fig_scene_control_pairs")

    print("\n" + "=" * 78)
    print("OUTPUT FILES")
    print("=" * 78)
    from PIL import Image
    for path in pa + pb:
        dims = ""
        if path.suffix == ".png":
            with Image.open(path) as im:
                dims = f"  {im.size[0]}x{im.size[1]} px"
        print(f"  {path}  {path.stat().st_size / 1024:8.1f} KB{dims}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
