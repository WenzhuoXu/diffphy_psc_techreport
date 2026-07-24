#!/usr/bin/env python3
"""Print every quantity and every verbatim string used in the critic figures.

Idempotent and read-only. Run it to re-check that the text inside
fig_critic_pipeline.tex and fig_plan_example.tex still matches the source files.

    python3 /Users/wenzhuox/diffphy_psc/techreport/figs/fig_values.py
"""
import json
from pathlib import Path

GOLD = Path("/Users/wenzhuox/diffphy_exp013/artifacts/runs/exp013/gold_v1/gold_core_v1.json")
PLAN = Path("/Users/wenzhuox/diffphy_exp013/scripts/exp013/exp029/design/planner_v3_verbatim.json")
ITEM = "3435a629-af23-50b3-bc74-8b34085e958d"


def main() -> int:
    gold = json.loads(GOLD.read_text())
    clips = gold["clips"]
    clip = next(c for c in clips if c["item_id"] == ITEM)

    print(f"source: {GOLD}")
    print(f"  frozen core clips                 = {len(clips)}")
    print(f"  frozen core human flaws           = {sum(len(c['flaws']) for c in clips)}")
    print()
    print("worked-example clip (the mousetrap prompt), from the frozen core:")
    print(f"  prompt (verbatim)                 = {clip['prompt']!r}")
    for fl in clip["flaws"]:
        print(f"  human-marked prompt span          = {fl['span']!r}")
        print(f"  human rationale (verbatim)        = {fl['reasoning']!r}")
        print(f"  human severity / category         = {fl['severity']} / {fl['category']}")
    print()

    plan = json.loads(PLAN.read_text())["mousetrap"]
    claims, lines, typed = plan["claims"], plan["raw_lines"], plan["typed"]
    n_sub = sum(len(v) for v in typed.values())
    print(f"source: {PLAN}  (verbatim planner output, no video read)")
    print(f"  claims written for this prompt     = {len(claims)}")
    print(f"  sub-checks across all claims       = {n_sub}")
    print(f"  claims with more than one check    = {sum(1 for v in typed.values() if len(v) > 1)}"
          f"  -> {[k for k, v in typed.items() if len(v) > 1]}")
    print(f"  composite claims recorded          = {list(plan['composite'])}")
    print()
    for cid in ("c2", "c4"):
        print(f"  claim {cid} text  = {claims[int(cid[1:])]!r}")
        print(f"  claim {cid} line  = {lines[cid]!r}")
        print(f"  claim {cid} kinds = {typed[cid]}")
    print()
    print("strings that must appear verbatim in fig_plan_example.tex:")
    print("  " + lines["c2"])
    print("  " + lines["c4"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
