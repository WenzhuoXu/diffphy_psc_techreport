#!/usr/bin/env python3
"""verify_report.py — does every number in the BUILT PDF still match the scripts?

WHY THIS IS THE LAST GATE. The acceptance test for this work is "every number was recomputed by
a script from per-item rows on disk". The generated tables satisfy that by construction, and a
separate check stops the PROSE from hardcoding a figure the tables own. What neither covers is
the built artefact: `main.pdf` is what a reader sees, and it can lag the rows by one forgotten
rebuild. A table that is correct in `tab_paired.tex` and stale in the PDF is still a wrong paper.

So this reads the delivered PDF's text and asserts the load-bearing figures appear in it, at the
values the scorers produce right now. It fails loudly on a mismatch instead of reporting a
percentage of checks passed, because "9 of 10 numbers agree" is not a state anything should ship
in.

    python analysis/verify_report.py            # exit 1 if the PDF disagrees with the rows
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PDF = HERE.parent / "main.pdf"
PAIRED = Path("/tmp/paired.json")
FROZEN = Path("/tmp/paired_frozen.json")
ABLATION = Path("/tmp/ablation.json")


def pdf_text() -> str:
    r = subprocess.run(["pdftotext", str(PDF), "-"], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"[FAIL] could not read {PDF}: {r.stderr[:200]}")
    # pdftotext line-wraps; collapse whitespace so "179/295" survives a line break
    return re.sub(r"\s+", " ", r.stdout)


def main() -> int:
    if not PDF.exists():
        raise SystemExit(f"[FAIL] {PDF} does not exist — build before verifying")
    txt = pdf_text()
    checks: list[tuple[str, str, bool]] = []

    def want(label: str, needle: str) -> None:
        checks.append((label, needle, needle in txt))

    if PAIRED.exists():
        p = json.loads(PAIRED.read_text())
        g, f, n = p["caught"]["graph"], p["caught"]["flat"], p["flaws"]
        lo, hi = p["ci"]
        want("paired: plan-based flaws", str(g))
        want("paired: per-claim flaws", str(f))
        want("paired: denominator", str(n))
        want("paired: clips", str(p["clips"]))
        want("paired: rate (graph)", f"{100.0*g/n:.1f}%")
        want("paired: rate (flat)", f"{100.0*f/n:.1f}%")
        want("paired: interval", f"[+{lo}, +{hi}]" if lo > 0 else f"[{lo:+d}, {hi:+d}]")
        want("paired: overlap graph-only", str(p["overlap"]["graph_only"]))
        want("paired: overlap flat-only", str(p["overlap"]["flat_only"]))
    else:
        print("[warn] /tmp/paired.json absent — paired figures unverified")

    if ABLATION.exists():
        rows = json.loads(ABLATION.read_text())["rows"]
        want("ablation: flaws (full)", f"{rows[0]['full_caught']}/{rows[0]['flaws']}")
        want("ablation: clips", str(rows[0]["clips"]))
        for r in rows:                       # EVERY row, so a new one cannot go unverified
            want(f"ablation[{r['env']}]: flaws off", f"{r['off_caught']}/{r['flaws']}")
            want(f"ablation[{r['env']}]: rate off",
                 f"{100.0*r['off_caught']/r['flaws']:.1f}%")
    else:
        print("[warn] /tmp/ablation.json absent — ablation figures unverified")

    if FROZEN.exists():
        z = json.loads(FROZEN.read_text())
        want("claims-matched: plan-based", f"{z['caught']['graph']}/{z['flaws']}")
        want("claims-matched: per-claim", f"{z['caught']['flat']}/{z['flaws']}")
        want("claims-matched: clips", str(z["clips"]))
    else:
        print("[warn] /tmp/paired_frozen.json absent — claims-matched figures unverified")

    # the published-run figures the descriptive tables rest on
    for label, needle in (("published: found/total", "228"),
                          ("published: denominator", "304"),
                          ("published: omitted clip", "fec3507b")):
        want(label, needle)

    bad = [(l, s) for l, s, ok in checks if not ok]
    for label, needle, ok in checks:
        print(f"  {'OK  ' if ok else 'FAIL'} {label:32s} expects {needle!r}")
    print(f"\n{len(checks) - len(bad)}/{len(checks)} figures present in the built PDF")
    if bad:
        print("[FAIL] the built PDF disagrees with the current rows — REBUILD, then re-verify:")
        for label, needle in bad:
            print(f"   {label}: {needle!r} not found")
        return 1
    print("[OK] every load-bearing figure in the PDF matches what the scorers produce now")
    return 0


if __name__ == "__main__":
    sys.exit(main())
