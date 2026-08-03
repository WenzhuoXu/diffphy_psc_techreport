#!/usr/bin/env python3
"""parse_results_page.py — turn the published EXP-029 results page into scoreable rows.

WHY THIS EXISTS. The headline numbers of the current framework (228 of 304 human flaws,
75%) live in one published HTML page; the per-clip JSONL rows that produced it are on the
pod that built it and are not in any repository. Rather than retype the numbers into LaTeX
-- which would make every figure in the report unverifiable -- this parses the page back
into one row per (clip, human flaw) with its category, our finding, and how the flaw was
scored, and RECOMPUTES the published totals from those rows. If a recomputed total
disagrees with the page's own banner, that is a hard error, not a warning: it means the
page and the parse disagree about what was measured and neither can be trusted.

The page is the artefact of record for this run. Every downstream table states that.

    python parse_results_page.py                       # fetch (cached), parse, verify
    python parse_results_page.py --out rows.jsonl      # also write the rows

Emits per flaw: clip, category, human sentence, our finding, state (caught|partial|missed),
whether the stronger reviewing model overruled a pass, and whether a tool measurement is
quoted on that finding.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import urllib.request
from pathlib import Path

URL = ("https://jiuxiang.s3.us-west-2.amazonaws.com/interns/wenzhuo/html/"
       "exp029_results/results.html")
CACHE = Path("/tmp/exp029_results.html")

# The page's own banner. Parsed out and checked against what we recompute, so a silent
# change to the page cannot slip through as a different number in the report.
BANNER = re.compile(r">(\d+)/(\d+)<[^>]*>\s*</div>\s*<div class=l>human flaws found")


def fetch(refresh: bool = False) -> str:
    if CACHE.exists() and not refresh:
        return CACHE.read_text()
    with urllib.request.urlopen(URL, timeout=120) as r:
        s = r.read().decode("utf-8", "replace")
    CACHE.write_text(s)
    return s


def _txt(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s)).strip()


def parse(page: str) -> tuple[list[dict], dict]:
    """One row per (clip, human flaw). Also returns the page's own stated totals."""
    # Banner tiles are <div class=n>VALUE</div><div class=l>LABEL</div>. Every pattern here
    # MUST match: a banner figure that silently fails to parse would skip its own
    # cross-check, which is how a gate stops being a gate.
    stated = {}
    tile = r"<div class=n>{v}</div><div class=l>{lab}"
    for label, pat in (
            ("caught_total", tile.format(v=r"(\d+)/(\d+)", lab="human flaws found")),
            ("full",         tile.format(v=r"(\d+)", lab="fully caught")),
            ("partial",      tile.format(v=r"(\d+)", lab="partial")),
            ("missed",       tile.format(v=r"(\d+)", lab="missed")),
            ("clips_full",   tile.format(v=r"(\d+)/(\d+)", lab="clips fully"))):
        m = re.search(pat, page, re.I | re.S)
        if not m:
            raise SystemExit(f"[FAIL] banner figure {label!r} not found in the page — the "
                             "layout changed; fix the pattern rather than skipping the check")
        stated[label] = (tuple(int(g) for g in m.groups()) if len(m.groups()) > 1
                         else int(m.group(1)))

    # the category table the annotators' own labels produce
    cat_table = {}
    for m in re.finditer(r"<tr><td>(\w+)</td><td class=n>(\d+)/(\d+)</td>"
                         r"<td class=n>(\d+)</td><td class=n>(\d+)</td>"
                         r"<td class=n>(\d+)</td>", page):
        c, found, tot, caught, part, miss = m.groups()
        cat_table[c] = dict(found=int(found), total=int(tot), caught=int(caught),
                            partial=int(part), missed=int(miss))
    if len(cat_table) < 8:
        raise SystemExit(f"[FAIL] parsed only {len(cat_table)} category rows; the page lists "
                         "ten. Fix the pattern rather than checking a subset.")

    rows: list[dict] = []
    # each clip is one .cc card; split on the card open tag so a card's text is self-contained
    cards = re.split(r"<div class='cc'", page)[1:]
    for card in cards:
        cid = (re.search(r"class='cid'>([0-9a-f]+)<", card) or [None, None])[1]
        st = (re.search(r"data-st='(\w+)'", card) or [None, None])[1]
        prompt = _txt((re.search(r"class='cp'[^>]*>(.*?)</div>", card, re.S)
                       or [None, ""])[1])
        for fl in re.findall(r"<div class='fl'>(.*?)</div>\s*(?=<div class='fl'>|$)",
                             card, re.S):
            cat = _txt((re.search(r"human annotator · (\w+)", fl) or [None, ""])[1]) \
                or (re.search(r"human annotator · (\w+)", fl) or [None, ""])[1]
            human = _txt((re.search(r"class='hum'>(.*?)</div>", fl, re.S) or [None, ""])[1])
            ours_m = re.search(r"class='ours( p)?'>(.*?)</div>", fl, re.S)
            ours = _txt(ours_m.group(2)) if ours_m else ""
            if "partial —" in fl or "partial &mdash;" in fl or (ours_m and ours_m.group(1)):
                state = "partial"
            elif "we caught it" in fl:
                state = "caught"
            else:
                state = "missed"
            if not human:
                continue
            rows.append(dict(
                clip=cid, clip_state=st, category=cat, prompt=prompt,
                human=human, ours=ours, state=state,
                reviewer_override=("overruled a pass" in fl),
                has_measurement=bool(re.search(r"\d+\.\d+|px/frame|radi|frames?\b", ours)),
            ))
    return rows, {"stated": stated, "cat_table": cat_table}


def verify(rows: list[dict], meta: dict) -> None:
    """Recompute the page's banner from the rows. Disagreement is fatal."""
    st = meta["stated"]
    got_caught = sum(1 for r in rows if r["state"] == "caught")
    got_partial = sum(1 for r in rows if r["state"] == "partial")
    got_missed = sum(1 for r in rows if r["state"] == "missed")
    total = len(rows)
    print(f"[parse] {total} (clip, flaw) rows over "
          f"{len({r['clip'] for r in rows})} clips")
    print(f"[parse] recomputed: caught {got_caught}  partial {got_partial}  "
          f"missed {got_missed}")
    print(f"[page ] stated:     caught {st.get('full')}  partial {st.get('partial')}  "
          f"missed {st.get('missed')}  found {st.get('caught_total')}")
    bad = []
    for k, got in (("full", got_caught), ("partial", got_partial), ("missed", got_missed)):
        want = st.get(k)
        if want is not None and want != got:
            bad.append(f"{k}: page says {want}, rows give {got}")
    if st.get("caught_total"):
        want_found, want_tot = st["caught_total"]
        if want_tot != total:
            bad.append(f"denominator: page says {want_tot} flaws, rows give {total}")
        if want_found != got_caught + got_partial:
            bad.append(f"found: page says {want_found}, rows give "
                       f"{got_caught + got_partial}")
    # the category table must also reproduce
    for c, want in (meta["cat_table"] or {}).items():
        sub = [r for r in rows if r["category"] == c]
        got = dict(total=len(sub),
                   caught=sum(1 for r in sub if r["state"] == "caught"),
                   partial=sum(1 for r in sub if r["state"] == "partial"),
                   missed=sum(1 for r in sub if r["state"] == "missed"))
        for k in ("total", "caught", "partial", "missed"):
            if want[k] != got[k]:
                bad.append(f"category {c}.{k}: page {want[k]} vs rows {got[k]}")
    if bad:
        print("\n[FAIL] the parse and the page disagree — the rows are NOT usable:")
        for b in bad:
            print("   ", b)
        raise SystemExit(2)
    print("[parse] OK — every published figure reproduces from the parsed rows")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    page = fetch(a.refresh)
    rows, meta = parse(page)
    verify(rows, meta)
    if a.out:
        with open(a.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"[write] {a.out}  ({len(rows)} rows)")
        p = Path(a.out).with_suffix(".meta.json")
        p.write_text(json.dumps(meta, indent=2))
        print(f"[write] {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
