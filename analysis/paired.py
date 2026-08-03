#!/usr/bin/env python3
"""paired.py — the head-to-head table: plan-based (graph) vs per-claim (flat) on the frozen 306.

WHAT MAKES THIS THE DECISIVE TABLE, and why the report's existing one is not. The pinned
question is whether orchestrating checks as a typed plan beats verifying each claim
one-by-one **at equal or lower cost**. The report's current table compares 197/286 to
179/286: a different denominator from the frozen set, and the plan-based side spent MORE
calls per clip (13.0 vs 12.1), so it never tested the equal-cost condition at all. A
confidence interval on the recall gain does not repair a cost mismatch.

So this scores both conditions:
  * on the SAME clips, from rows written this run (no historical rows, no published page),
  * against the SAME frozen 306 human flaws, with the denominator printed,
  * on the SAME frozen claim decompositions -- `runners/freeze_claims.py` writes one
    decomposition per prompt and both conditions read it, so neither can be handed easier
    claims than the other,
  * with the paired overlap exposed: caught by both, graph-only, flat-only, neither. A
    net difference of zero can still hide a large disagreement in both directions, and that
    disagreement is the interesting part.

Bootstrap resamples CLIPS, not flaws, because flaws within one clip are dependent, and both
conditions are resampled on the SAME sampled clips so the interval is paired. The interval
covers sampling uncertainty only -- it says nothing about scorer nondeterminism, which needs
repeated control rescores.

    python paired.py --graph graph_paired.jsonl --flat flat_baseline_result.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

RESULTS = Path("~/diffphy_psc/artifacts/runs/exp033").expanduser()
GOLD = Path("~/diffphy_exp013/artifacts/runs/exp013/gold_v1/gold_core_v1.json").expanduser()
SEED = 20260803
BOOT = 10000


def load(path: Path) -> dict[str, dict]:
    """clip_id -> row. A row marked errored is NOT a score of zero; it is excluded and
    counted, because scoring a failed clip as a miss would flatter whichever condition
    happened to crash less."""
    rows, errored, dup_disagree = {}, [], []
    if not path.exists():
        return rows
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("errored"):
            errored.append(r["clip"])
            continue
        # Two workers on disjoint orders can both reach one clip. Last-wins is only safe if the
        # duplicate rows AGREE, so check rather than assume: a silent disagreement would make
        # the score depend on file order. Measured on this run: 7 duplicated clips, all
        # identical, i.e. the condition is deterministic on repeat.
        prev = rows.get(r["clip"])
        if prev is not None:
            a = sorted(prev.get("covered_flaws") or [])
            b = sorted(r.get("covered_flaws") or [])
            if a != b:
                dup_disagree.append((r["clip"], a, b))
        rows[r["clip"]] = r
    if dup_disagree:
        print(f"[load] {path.name}: !! {len(dup_disagree)} duplicated clip(s) DISAGREE on "
              f"covered flaws; the score would depend on file order: "
              f"{[c[:8] for c, _, _ in dup_disagree][:5]}")
    if errored:
        print(f"[load] {path.name}: {len(rows)} scored, {len(errored)} errored and EXCLUDED "
              f"({[c[:8] for c in errored][:6]})")
    else:
        print(f"[load] {path.name}: {len(rows)} scored, 0 errored")
    return rows


def calls(r: dict) -> int:
    c = r.get("cost") or {}
    # Every model/tool invocation the condition spent, EXCLUDING the gold-matching calls:
    # matching is the scorer, not the critic, and charging it would penalise whichever
    # condition alleged more.
    return sum(v for k, v in c.items()
               if k.endswith("_calls") and k != "match_calls" and isinstance(v, (int, float)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="graph_paired.jsonl")
    ap.add_argument("--flat", default="flat_baseline_result.jsonl")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    gold = {c["item_id"]: c for c in json.load(open(GOLD))["clips"]}
    G = load(RESULTS / a.graph)
    F = load(RESULTS / a.flat)
    both = sorted(set(G) & set(F))
    if not both:
        print("\n[paired] NO clip has been scored by both conditions yet — nothing to report.")
        print(f"  graph {len(G)} clips, flat {len(F)} clips")
        return 0

    n_flaws = sum(len(gold[c]["flaws"]) for c in both)
    print(f"\n=== paired on {len(both)} clips both conditions scored "
          f"({n_flaws} human flaws) ===")
    print(f"    graph has {len(G)} clips, flat has {len(F)}; "
          f"{len(set(G) ^ set(F))} scored by only one and excluded")

    def caught(R, c):
        return set(R[c].get("covered_flaws") or [])

    tot = {"graph": 0, "flat": 0}
    ov = collections.Counter()
    for c in both:
        g, f = caught(G, c), caught(F, c)
        allg = {x["flaw_uid"] for x in gold[c]["flaws"]}
        tot["graph"] += len(g)
        tot["flat"] += len(f)
        ov["both"] += len(g & f)
        ov["graph_only"] += len(g - f)
        ov["flat_only"] += len(f - g)
        ov["neither"] += len(allg - g - f)

    # Recall alone rewards over-flagging, so the allegation volume and the share of
    # allegations that land must sit next to it. A condition that accuses more gets more
    # chances at the matcher; without these columns a volume effect reads as an accuracy win.
    alleg = {}
    for k in ("graph", "flat"):
        R = G if k == "graph" else F
        n_al = sum(len(R[c].get("alleged") or []) for c in both)
        n_hit = sum(1 for c in both for a in (R[c].get("alleged") or [])
                    if a.get("matched_flaw"))
        alleg[k] = dict(alleged=n_al, landed=n_hit,
                        per_clip=round(n_al / len(both), 2),
                        precision=round(100.0 * n_hit / n_al, 1) if n_al else 0.0)

    for k in ("graph", "flat"):
        R = G if k == "graph" else F
        cps = [calls(R[c]) for c in both]
        lab = "plan-based (graph)" if k == "graph" else "per-claim (flat)"
        al = alleg[k]                     # NOT `a`: that is the argparse namespace
        print(f"  {lab:22s} caught {tot[k]:3d}/{n_flaws} = {100*tot[k]/n_flaws:4.1f}%   "
              f"calls/clip {sum(cps)/len(cps):5.1f}   "
              f"allegations/clip {al['per_clip']:4.1f}   landed {al['precision']:4.1f}%")
    print(f"  overlap: both {ov['both']}  graph-only {ov['graph_only']}  "
          f"flat-only {ov['flat_only']}  neither {ov['neither']}")
    print(f"  net difference: {tot['graph'] - tot['flat']:+d} flaws")

    # paired bootstrap over clips: same resampled clips for both conditions
    rng = random.Random(SEED)
    d = []
    for _ in range(BOOT):
        s = [both[rng.randrange(len(both))] for _ in both]
        d.append(sum(len(caught(G, c)) for c in s) - sum(len(caught(F, c)) for c in s))
    d.sort()
    lo, hi = d[int(0.025 * BOOT)], d[int(0.975 * BOOT)]
    print(f"  paired 95% interval on (graph - flat): [{lo:+d}, {hi:+d}]  "
          f"(resampling clips, seed {SEED}, {BOOT} resamples)")
    verdict = ("graph ahead" if lo > 0 else "flat ahead" if hi < 0
               else "NOT SEPARABLE — the interval covers zero")
    print(f"  verdict: {verdict}")

    res = dict(clips=len(both), flaws=n_flaws, caught=tot, overlap=dict(ov),
               allegations=alleg,
               net=tot["graph"] - tot["flat"], ci=[lo, hi], verdict=verdict,
               calls_per_clip={k: round(sum(calls((G if k == "graph" else F)[c])
                                            for c in both) / len(both), 2)
                               for k in ("graph", "flat")},
               seed=SEED, bootstrap=BOOT)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2))
        print(f"\n[write] {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
