#!/usr/bin/env python3
"""headline_cost.py — cost per clip for the headline run, from that run's own rows.

WHY A SCRIPT AND NOT A TYPED NUMBER. Every other figure in the Results section is
regenerated on each build and checked against the delivered PDF. The cost cell of the
external-comparison table was the one blank our own row had, and typing it in would have made
it the only number in that section nothing verifies.

WHICH RUN. The headline (228 of 304) is `v3_physics_result.jsonl` in the exp029 tree. It is
identified, not assumed: it is the only candidate that overlaps all 149 page clips AND
contains the gpt-5.6-sol override records the page displays (115 of them). `v3_why_result` and
`v3_full150_result` overlap the same clips but carry no override, so neither produced that page.

WHAT IS COUNTED. Every model and tool invocation the critic spent -- planning, execution and
fallback verification -- and NOT the gold-matching calls, which belong to the scorer rather
than to the system under test. Charging them would penalise whichever system alleged more.

    python analysis/headline_cost.py            # print the figures
    python analysis/headline_cost.py --json out.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

ROWS = Path("~/diffphy_exp013/scripts/exp013/exp029/v3_physics_result.jsonl").expanduser()
PAGE_ROWS = Path("/tmp/e29_228.jsonl")          # written by parse_results_page.py --out


def calls(row: dict) -> int:
    c = row.get("cost") or {}
    return sum(v for k, v in c.items()
               if k.endswith("_calls") and k != "match_calls" and isinstance(v, (int, float)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    if not ROWS.exists():
        raise SystemExit(f"[FAIL] headline rows not found: {ROWS}")
    if not PAGE_ROWS.exists():
        raise SystemExit(f"[FAIL] {PAGE_ROWS} absent — run "
                         "parse_results_page.py --out /tmp/e29_228.jsonl first")

    page = {json.loads(l)["clip"] for l in open(PAGE_ROWS) if l.strip()}
    rows = [json.loads(l) for l in open(ROWS) if l.strip()]
    rows = [r for r in rows if r["clip"][:8] in page]

    # The override records are what identify this file as the run behind the page. Assert it,
    # so pointing this script at the wrong file fails loudly instead of reporting a cost for
    # some other run.
    n_override = sum(1 for r in rows if "gpt-5.6-sol" in json.dumps(r))
    if n_override == 0:
        raise SystemExit("[FAIL] these rows carry no gpt-5.6-sol override; they did not "
                         "produce the headline page")
    if len(rows) != len(page):
        raise SystemExit(f"[FAIL] {len(rows)} rows cover {len(page)} page clips")

    cs = [calls(r) for r in rows]
    ws = [(r.get("cost") or {}).get("wall_s", 0) for r in rows]
    out = dict(clips=len(rows), overrides=n_override,
               calls_mean=round(st.mean(cs), 1), calls_median=st.median(cs),
               calls_min=min(cs), calls_max=max(cs),
               wall_median_s=round(st.median(ws)))
    print(f"headline run: {out['clips']} clips, {out['overrides']} rows carrying an override")
    print(f"  calls per clip : mean {out['calls_mean']}  median {out['calls_median']}  "
          f"range {out['calls_min']}-{out['calls_max']}")
    print(f"  wall per clip  : median {out['wall_median_s']} s")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2))
        print(f"[write] {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
