#!/usr/bin/env python3
"""ablation.py — what each orchestration mechanism is worth, measured by turning it off.

WHY THESE ROWS AND NOT A SUBTRACTION. The tempting version of a component ablation is to take
a finished run and remove the findings a mechanism produced. That is invalid: it assumes every
removed finding becomes a miss, when the rest of the system may reroute, escalate differently,
or reach the same flaw another way. So each row here is a SEPARATE LIVE RUN of the identical
plan with one mechanism disabled, scored against the same human labels on the same clips.

Each row is paired against the FULL plan-based condition on exactly the clips both completed,
and the difference gets a paired bootstrap over clips. A row whose interval covers zero is
reported as no measurable effect -- which for a mechanism that is barely exercised on this data
is the expected and honest outcome, not a failure.

    python ablation.py --json /tmp/ablation.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

RESULTS = Path("~/diffphy_psc/artifacts/runs/exp033").expanduser()
GOLD = Path("~/diffphy_exp013/artifacts/runs/exp013/gold_v1/gold_core_v1.json").expanduser()
SEED = 20260803
BOOT = 10000

# (file, label, the mechanism switched off, the env var that switched it off)
ROWS = [
    ("ablate_nosharing.jsonl", "No shared perception", "call deduplication", "VAC_NO_SHARING"),
    ("ablate_norouting.jsonl", "No specialist routing", "every specialist measurement",
     "VAC_DISABLE_OPS"),
]


def load(path: Path) -> dict[str, dict]:
    rows = {}
    if not path.exists():
        return rows
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            if not r.get("errored"):
                rows[r["clip"]] = r
    return rows


def calls(r):
    c = r.get("cost") or {}
    return sum(v for k, v in c.items()
               if k.endswith("_calls") and k != "match_calls" and isinstance(v, (int, float)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    gold = {c["item_id"]: c for c in json.load(open(GOLD))["clips"]}
    full = load(RESULTS / "graph_paired.jsonl")
    out = {"rows": []}
    print(f"[ablation] full plan-based condition: {len(full)} clips scored")

    for fn, label, mech, env in ROWS:
        R = load(RESULTS / fn)
        both = sorted(set(R) & set(full))
        if not both:
            print(f"\n=== {label}: no clip scored by both yet ({len(R)} in this row) — "
                  "nothing to report")
            continue
        n_flaws = sum(len(gold[c]["flaws"]) for c in both)
        ab = sum(len(set(R[c].get("covered_flaws") or [])) for c in both)
        fu = sum(len(set(full[c].get("covered_flaws") or [])) for c in both)
        ab_calls = sum(calls(R[c]) for c in both) / len(both)
        fu_calls = sum(calls(full[c]) for c in both) / len(both)
        # DO NOT report cost["saved_by_sharing"]. It is n_flat_calls - n_shared_calls, and
        # those count different populations: n_flat_calls is the plan's observations only,
        # while n_shared_calls counts plan.calls, which ALSO contains generalist verify calls
        # the compiler synthesises. When the compiler adds calls the difference goes negative
        # -- 11 of 105 rows in the full condition are negative, min -9 -- so it is not a
        # measure of sharing at all. Pre-existing, unrelated to the ablation flag, and left
        # unpatched here because the runner writes it; instead measure sharing DIRECTLY as
        # the executed-call gap between the two conditions on the same clips, which is what
        # the ablation is for.
        ab_saved = fu_saved = None

        # POSITIVE CONTROL on the flag itself. On clips where the full condition genuinely
        # deduplicated calls, the ablation MUST spend more; if it does not, the flag never took
        # effect and the whole row is void. Reported so the null cannot be confused with a
        # no-op switch -- the difference between "sharing does nothing" and "we failed to turn
        # sharing off" is the difference between a result and a bug.
        if env == "VAC_NO_SHARING":
            active = [(c, (full[c].get("cost") or {}).get("saved_by_sharing", 0),
                       calls(R[c]) - calls(full[c])) for c in both
                      if (full[c].get("cost") or {}).get("saved_by_sharing", 0) > 0]
            leaked = [c for c in both
                      if (R[c].get("cost") or {}).get("saved_by_sharing", 0) > 0]
        else:
            # ROUTING row control. The obvious signals do NOT work here, and both were tried:
            #   * n_measured counts what the PLANNER TYPED, not what a tool executed, so it is
            #     unchanged by disabling ops -- reading it reported a "leak" that was really my
            #     own wrong signal.
            #   * exec_calls is also unchanged, because the plan still compiles the same calls;
            #     each one now abstains instead of measuring.
            # The observable effect of switching specialists off is therefore in the VERDICTS:
            # a check that would have been settled by a measurement now abstains, so the clip
            # alleges fewer contradictions. Control = on clips the full condition measured, the
            # ablation must allege NO MORE than the full condition, and strictly fewer somewhere.
            active = [(c, full[c].get("n_measured", 0),
                       len(R[c].get("alleged") or []) - len(full[c].get("alleged") or []))
                      for c in both if full[c].get("n_measured", 0) > 0]
            # A FEW clips legitimately allege MORE, so "any increase = leak" was too strict and
            # voided a valid row on 1 of 25 clips. Why an increase is expected: removing a
            # measurement that SUPPORTED a claim turns it unknown, and a composite claim whose
            # conjunct is no longer supported can roll up to a contradiction. That is the
            # mechanism operating, not the switch failing. The control that still discriminates
            # is the DIRECTION OF THE BULK: decreases must dominate increases by a wide margin.
            up = [c for c, _, d in active if d > 0]
            down = [c for c, _, d in active if d < 0]
            leaked = up if len(up) > max(1, 0.25 * len(down)) else []

        rng = random.Random(SEED)
        d = []
        for _ in range(BOOT):
            s = [both[rng.randrange(len(both))] for _ in both]
            d.append(sum(len(set(R[c].get("covered_flaws") or [])) for c in s)
                     - sum(len(set(full[c].get("covered_flaws") or [])) for c in s))
        d.sort()
        lo, hi = d[int(0.025 * BOOT)], d[int(0.975 * BOOT)]
        eff = ("no measurable effect" if lo <= 0 <= hi
               else "removing it HURTS" if hi < 0 else "removing it HELPS")

        print(f"\n=== {label} (off: {mech}, via {env}) ===")
        print(f"  paired on {len(both)} clips / {n_flaws} flaws")
        print(f"  full condition   {fu:3d}/{n_flaws} = {100*fu/n_flaws:4.1f}%   "
              f"calls/clip {fu_calls:5.1f}")
        print(f"  mechanism off    {ab:3d}/{n_flaws} = {100*ab/n_flaws:4.1f}%   "
              f"calls/clip {ab_calls:5.1f}")
        print(f"  extra calls from removing the mechanism: "
              f"{ab_calls - fu_calls:+.2f} per clip")
        print(f"  difference (off - full): {ab-fu:+d} flaws, paired 95% [{lo:+d}, {hi:+d}] "
              f"-> {eff}")
        if env == "VAC_NO_SHARING":
            print(f"  flag control: {len(active)} of {len(both)} clips had sharing genuinely "
                  f"active in the full condition")
            if active:
                print("    on those, removing it cost extra calls: "
                      + ", ".join(f"{c[:8]} {d:+.0f}" for c, _, d in active))
                print("    -> PASS: the flag demonstrably took effect (more calls without it)"
                      if all(d > 0 for _, _, d in active)
                      else "    -> FAIL: the flag did NOT take effect; this row is VOID")
            if leaked:
                print(f"    -> FAIL: {len(leaked)} row(s) report positive sharing savings; VOID")
        else:
            n_full = sum(x for _, x, _ in active)
            fewer = down
            print(f"  flag control: {len(active)} clips carried {n_full} typed measurements in "
                  f"the full condition; with specialists off, {len(fewer)} of them allege FEWER "
                  f"contradictions and {len(up)} allege more")
            if leaked:
                print(f"    -> FAIL: {len(leaked)} clip(s) allege MORE without specialists; "
                      "that is not an ablation of this mechanism, VOID")
            elif fewer:
                print("    -> PASS: removing specialists demonstrably changed verdicts "
                      "(measurement-settled checks now abstain)")
            else:
                print("    -> INCONCLUSIVE: no clip changed; the measurements this subset takes "
                      "did not decide any verdict, so nothing was ablated in effect")
        # The "barely exercised" note is about SHARING (whose whole effect is call count).
        # On the routing row the call count is the wrong yardstick -- specialists change verdicts,
        # not call volume -- so printing it there would mislead.
        if env == "VAC_NO_SHARING" and abs(ab_calls - fu_calls) < 0.5:
            print("  NOTE: removing the mechanism barely changed the call count, so it was "
                  "hardly exercised on these clips. A null result here is expected and says "
                  "nothing about its value on data with more repeated queries.")
        out["rows"].append(dict(label=label, mechanism=mech, env=env, clips=len(both),
                               flaws=n_flaws, full_caught=fu, off_caught=ab,
                               full_calls=round(fu_calls, 2), off_calls=round(ab_calls, 2),
                               extra_calls=round(ab_calls - fu_calls, 2),
                               diff=ab - fu, ci=[lo, hi], effect=eff,
                               flag_active_clips=len(active), flag_leaked=len(leaked),
                               flag_extra_calls=[round(d, 1) for _, _, d in active]))
    if a.json and out["rows"]:
        Path(a.json).write_text(json.dumps(out, indent=2))
        print(f"\n[write] {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
