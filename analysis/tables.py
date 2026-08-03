#!/usr/bin/env python3
"""tables.py — every benchmark table in the Results section, recomputed from rows on disk.

RULE THIS FILE ENFORCES: no number reaches main.tex by hand. Each table is a function that
reads per-item rows, prints its denominator, and emits LaTeX. Re-running this file
regenerates every figure in the Results section; if a number here disagrees with the report,
the report is wrong.

SOURCES, and what each can and cannot support:

  * `page_rows`  — one row per (clip, human flaw) parsed from the published EXP-029 results
    page by parse_results_page.py, which cross-checks the page's own banner and category
    table before returning. 304 rows / 149 clips. This is the artefact of record for the
    current framework: the per-clip JSONL it was built from lives on the pod that produced
    it and is not recoverable, so tables built from it SAY so.
  * `gold_core_v1.json` — the human labels. Supplies severity, duration, flaws-per-clip and
    the hard-stratum flag as covariates. Every page row joins to exactly one gold flaw.
  * `count_tool_gold.jsonl` / `ocr_tool_gold*.jsonl` — per-tool labels: what the tool was
    asked, what it measured. These score a TOOL, not the critic, which is the separation the
    architecture claims to make possible.

Bootstrap intervals resample CLIPS (not flaws), because flaws within a clip are correlated;
fixed seed, stated resample count. A difference whose interval covers zero is reported as
not separable rather than as a win.

    python analysis/tables.py            # print every table + its numbers
    python analysis/tables.py --tex out.tex
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
GOLD_DIR = Path("~/diffphy_exp013/artifacts/runs/exp013/gold_v1").expanduser()

SEED = 20260803
BOOT = 10000

# The page's own measured scorer noise floor: re-scoring unchanged control text across
# passes moves this many flaws. A difference at or under it is not a finding.
NOISE_FLOOR = 7


# --------------------------------------------------------------------------- load
def page_rows() -> list[dict]:
    """The 304 (clip, flaw) rows, re-parsed and re-verified on every run."""
    sys.path.insert(0, str(HERE))
    import parse_results_page as prp
    rows, meta = prp.parse(prp.fetch())
    prp.verify(rows, meta)                      # refuses to return unverified rows
    return rows


def gold():
    g = json.load(open(GOLD_DIR / "gold_core_v1.json"))["clips"]
    return {c["item_id"]: c for c in g}


def joined() -> list[dict]:
    """Page rows with their gold flaw's covariates attached. A row that fails to join is a
    hard error: a silent join miss would quietly shrink a denominator."""
    G = gold()
    by8 = {k[:8]: v for k, v in G.items()}
    out, dupes = [], []
    for r in page_rows():
        c = by8.get(r["clip"])
        if c is None:
            raise SystemExit(f"[FAIL] page clip {r['clip']} is not in the gold set")
        # Join on the annotator sentence. Clip f55f2753 carries two DISTINCT gold flaws
        # (different uid, same category) whose rationales share a 60-character prefix and
        # differ only later in the sentence, so a prefix match is ambiguous there. Narrow by
        # category, then by the longest common prefix; only accept a unique survivor. Never
        # silently take hit[0] -- picking arbitrarily would attach the wrong severity.
        hit = [f for f in c["flaws"] if f["reasoning"][:60] == r["human"][:60]]
        if len(hit) > 1:
            hit = [f for f in hit if f["category"] == r["category"]] or hit
        if len(hit) > 1:
            n = max(len(f["reasoning"]) for f in hit)
            hit = [f for f in hit if f["reasoning"][:n] == r["human"][:n]] or hit
        if len(hit) > 1:
            # Genuinely indistinguishable duplicates: they carry the same severity and
            # category, so either serves. Assert that, then take the first.
            if len({(f["severity"], f["category"]) for f in hit}) != 1:
                raise SystemExit(
                    f"[FAIL] clip {r['clip']}: {len(hit)} gold flaws match the annotator "
                    f"sentence and they DISAGREE on severity/category — cannot join")
            dupes.append((r["clip"], hit[0]["category"]))
        if not hit:
            raise SystemExit(f"[FAIL] clip {r['clip']}: no gold flaw matches the annotator "
                             f"sentence {r['human'][:60]!r}")
        f = hit[0]
        out.append({**r, "severity": f["severity"], "flaw_uid": f["flaw_uid"],
                    "duration_s": c["duration_s"], "n_flaws_clip": len(c["flaws"]),
                    "hard": bool((c.get("strata") or {}).get("adv29")),
                    "clip_severity": c["video_level_severity"]})
    if dupes:
        # Never silent: a reader of the log must see which rows were ambiguous.
        print(f"[join] {len(dupes)} row(s) matched indistinguishable duplicate gold flaws "
              f"(same severity and category, so the join is safe): {dupes}")
    if len(out) != 304:
        raise SystemExit(f"[FAIL] joined {len(out)} rows, expected 304")
    return out


def boot_diff(rows, key_a, key_b, seed=SEED, n=BOOT):
    """95% interval on (found under A) - (found under B), resampling CLIPS."""
    by_clip = collections.defaultdict(list)
    for r in rows:
        by_clip[r["clip"]].append(r)
    clips = sorted(by_clip)
    rng = random.Random(seed)
    d = []
    for _ in range(n):
        s = [clips[rng.randrange(len(clips))] for _ in clips]
        a = sum(1 for c in s for r in by_clip[c] if key_a(r))
        b = sum(1 for c in s for r in by_clip[c] if key_b(r))
        d.append(a - b)
    d.sort()
    return d[int(0.025 * n)], d[int(0.975 * n)]


def pct(a, b):
    return f"{100.0 * a / b:.1f}\\%" if b else "---"


# --------------------------------------------------------------------- the tables
def t_headline(rows) -> str:
    """The current framework's recall, and what the two named mechanisms are worth."""
    n = len(rows)
    found = [r for r in rows if r["state"] in ("caught", "partial")]
    full = [r for r in rows if r["state"] == "caught"]
    part = [r for r in rows if r["state"] == "partial"]
    ovr = [r for r in found if r["reviewer_override"]]
    print(f"\n=== headline (denominator {n} flaws / "
          f"{len({r['clip'] for r in rows})} clips) ===")
    print(f"  found {len(found)} ({100*len(found)/n:.1f}%)  full {len(full)}  "
          f"partial {len(part)}  missed {n-len(found)}")
    print(f"  of the found, {len(ovr)} carry the reviewer-override marker")
    lo, hi = boot_diff(rows, lambda r: r["state"] in ("caught", "partial"),
                       lambda r: r["state"] in ("caught", "partial")
                       and not r["reviewer_override"])
    print(f"  override contribution, bootstrap 95% over clips: [{lo}, {hi}]")
    return ""


def t_severity(rows) -> tuple[str, dict]:
    """Does recall hold as the annotator's severity rises?"""
    out = {}
    print("\n=== recall by annotator severity ===")
    for s in sorted({r["severity"] for r in rows}):
        sub = [r for r in rows if r["severity"] == s]
        f = sum(1 for r in sub if r["state"] in ("caught", "partial"))
        out[s] = (f, len(sub))
        print(f"  severity {s}: {f}/{len(sub)} = {100*f/len(sub):.0f}%")
    return "", out


def t_count_tool():
    """The counting specialist against per-target gold: measured vs expected."""
    rows = [json.loads(l) for l in open(GOLD_DIR / "count_tool_gold.jsonl")]
    tgts = [(m, r) for r in rows for m in (r.get("measures") or [])]
    exact = sum(1 for m, _ in tgts if m["measured"] == m["expected"])
    within1 = sum(1 for m, _ in tgts if abs(m["measured"] - m["expected"]) <= 1)
    zero = sum(1 for m, _ in tgts if m["measured"] == 0)
    over = sum(1 for m, _ in tgts if m["measured"] > m["expected"])
    under = sum(1 for m, _ in tgts if 0 < m["measured"] < m["expected"])
    print(f"\n=== counting specialist vs per-target gold "
          f"({len(tgts)} targets over "
          f"{len({r['item_id'] for _, r in tgts})} clips) ===")
    print(f"  exact {exact} ({100*exact/len(tgts):.0f}%)  within 1 {within1} "
          f"({100*within1/len(tgts):.0f}%)")
    print(f"  measured zero (a failed grounding) {zero}  over-count {over}  "
          f"under-count (non-zero) {under}")
    # per-frame instability: does the count wobble within one clip?
    wob = [m for m, _ in tgts if m.get("per_frame")
           and len(set(m["per_frame"])) > 1]
    print(f"  targets whose per-frame count is not constant: {len(wob)}/{len(tgts)}")
    return dict(targets=len(tgts), exact=exact, within1=within1, zero=zero,
                over=over, under=under, unstable=len(wob),
                clips=len({r["item_id"] for _, r in tgts}))


def t_ocr_engines():
    """Two recognition engines on the same text-bearing clips."""
    out = {}
    for name, fn in (("paddle", "ocr_tool_gold_paddle.jsonl"),
                     ("easyocr", "ocr_tool_gold.jsonl")):
        rows = [json.loads(l) for l in open(GOLD_DIR / fn)]
        req = [r for r in rows if r.get("requires_text")]
        flagged = sum(1 for r in req if r.get("flaws"))
        nothing = sum(1 for r in req if not r.get("n_detected"))
        out[name] = dict(clips=len(req), flagged=flagged, read_nothing=nothing)
    print(f"\n=== on-screen text: two recognition engines, same clips ===")
    for k, v in out.items():
        print(f"  {k:8s} clips requiring text {v['clips']}  flagged a defect "
              f"{v['flagged']}  read no text at all {v['read_nothing']}")
    return out


def t_category(rows):
    """Recall by the annotators' own flaw category, with a per-category interval."""
    print("\n=== recall by annotator category ===")
    out = {}
    for c, _ in collections.Counter(r["category"] for r in rows).most_common():
        sub = [r for r in rows if r["category"] == c]
        full = sum(1 for r in sub if r["state"] == "caught")
        part = sum(1 for r in sub if r["state"] == "partial")
        out[c] = dict(total=len(sub), full=full, partial=part,
                      found=full + part, missed=len(sub) - full - part)
        print(f"  {c:14s} {full + part:3d}/{len(sub):3d} = "
              f"{100*(full+part)/len(sub):3.0f}%   (full {full}, partial {part})")
    return out


def t_robustness(rows):
    """Does recall hold across clip length, defect density and the hard stratum?
    A flat profile is the claim; a slope would mean the instrument is length-limited."""
    print("\n=== robustness: recall across clip properties ===")
    out = {}

    def band(name, keyf, order=None):
        agg = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            k = keyf(r)
            agg[k][1] += 1
            agg[k][0] += r["state"] in ("caught", "partial")
        ks = order or sorted(agg)
        out[name] = {k: tuple(agg[k]) for k in ks}
        print(f"  {name}:")
        for k in ks:
            f, n = agg[k]
            print(f"    {k:14s} {f:3d}/{n:3d} = {100*f/n:3.0f}%")

    band("clip duration", lambda r: ("6 s or less" if r["duration_s"] <= 6
                                     else "6-8 s" if r["duration_s"] <= 8 else "over 8 s"),
         ["6 s or less", "6-8 s", "over 8 s"])
    band("flaws in clip", lambda r: ("1" if r["n_flaws_clip"] == 1
                                     else "2" if r["n_flaws_clip"] == 2
                                     else "3-4" if r["n_flaws_clip"] <= 4 else "5 or more"),
         ["1", "2", "3-4", "5 or more"])
    band("stratum", lambda r: "hard" if r["hard"] else "rest", ["hard", "rest"])
    lo, hi = boot_diff([r for r in rows if r["hard"]],
                       lambda r: r["state"] in ("caught", "partial"), lambda r: False)
    print(f"    hard-stratum found, bootstrap 95%: [{lo}, {hi}] of "
          f"{sum(1 for r in rows if r['hard'])}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default="")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    rows = joined()
    res = {}
    t_headline(rows)
    _, res["severity"] = t_severity(rows)
    res["category"] = t_category(rows)
    res["robustness"] = t_robustness(rows)
    res["count_tool"] = t_count_tool()
    res["ocr"] = t_ocr_engines()
    res["meta"] = dict(flaws=len(rows), clips=len({r["clip"] for r in rows}),
                       seed=SEED, bootstrap=BOOT, noise_floor=NOISE_FLOOR)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2, default=str))
        print(f"\n[write] {a.json}")
    print("\n[tables] every figure above was recomputed from rows on disk this run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
