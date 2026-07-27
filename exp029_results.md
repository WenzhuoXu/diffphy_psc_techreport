# EXP-029 — plan-based critic vs per-claim verification

All figures recomputed from raw per-clip rows by `analysis/techreport_results.py` and `analysis/techreport_failures.py`. Bootstrap CIs use pre-registered seed 20260724, 10,000 resamples.

## Headline

Paired clip by clip on **149 clips** — the 149 of the 150-clip evaluation core that both systems completed. Those clips carry 304 of the core's 306 flaw entries. Eighteen of the 304 are duplicates (12 clips list the same span two or three times under different ids), and the scoring loop credits one accusation to at most one flaw, so duplicates are uncatchable by construction; collapsing identical spans gives **286 distinct flaws**. (Core-wide the same collapse gives 288 distinct flaws of 306.) Both denominators are shown — the correction applies to every system equally and does not move the comparison.

| System | Flaws caught | Recall | As scored (with duplicates) | Calls/clip | Alleg. hit rate |
|---|---|---|---|---|---|
| Plan-based critic, all fixed specialists | 197 / 286 | **68.9%** | 197 / 304 (64.8%) | 13.0 | 40.1% |
| Per-claim verification, one claim at a time | 179 / 286 | **62.6%** | 179 / 304 (58.9%) | 12.1 | 40.2% |

**The plan-based critic catches 18 more flaws than per-claim verification** (95% CI [+8, +28]), at about 0.9 more calls per clip (~7% more). This is a trade of accuracy against cost, not a dominance result: no run holds the two systems to an equal call budget.

## Where the gain comes from

Counted against the **uncollapsed** flaw list (304 entries), so the per-category
totals here sum to 304 rather than 286; the duplicates fall almost entirely in
`action`. The miss-rate table further down uses the collapsed list.

| Flaw category | n | Plan-based | Per-claim |
|---|---|---|---|
| action | 137 | 98 (72%) | 89 (65%) |
| count | 36 | 18 (50%) | 16 (44%) |
| object | 30 | 18 (60%) | 18 (60%) |
| text_ocr | 25 | 19 (76%) | 15 (60%) |
| spatial | 19 | 13 (68%) | 11 (58%) |
| order_timing | 17 | 11 (65%) | 11 (65%) |
| attribute | 16 | 6 (38%) | 5 (31%) |
| camera_style | 10 | 5 (50%) | 5 (50%) |
| other | 7 | 5 (71%) | 4 (57%) |
| physics_motion | 7 | 4 (57%) | 5 (71%) |

The largest relative gain is rendered text, where a recognition-head OCR engine reads glyphs off the pixels. A generative reader is unusable there: given the same crop it reported fluent English that is not on the object, while the OCR head returned the actual `"RRVLE"`, `"ToeFan"`.

## Why the misses are missed

Of **89 uncaught flaws**:

| Failure mode | n | Share |
|---|---|---|
| Checked and cleared — a claim covered it, the critic said the video was fine | 79 | 88.8% |
| Flagged by the critic, not credited by the scoring protocol | 8 | 9.0% |
| Never decomposed — no claim covers the flaw | 2 | 2.2% |

**No miss is caused by a specialist returning a wrong measurement.** 76 of the 79 cleared flaws were decided by a bare yes/no judgement with no measurement attached; only 3 involved a measurement at all. Across the whole run, just **14.1%** of executed checks are measurements.

Within those bare judgements, roughly half are not judge errors: the claim that was checked is a *weaker* assertion than the one the human flagged, because decomposition dropped the discriminating detail — a stated quantity, a spatial qualifier, or the second half of a compound sentence ("the two people lose balance, **arms flailing, and tumble overboard**" was checked as "Two people lose balance"). The judge answers that weaker claim correctly; the video is wrong about the part that was discarded. The remainder are genuine judge misses on the full assertion.

### Miss rate by category

Counted against the **collapsed** flaw list (286 distinct flaws).

| Category | Uncaught / total | Miss rate |
|---|---|---|
| action | 36 / 134 | 26.9% |
| object | 12 / 30 | 40.0% |
| count | 8 / 26 | 30.8% |
| text_ocr | 4 / 23 | 17.4% |
| spatial | 5 / 18 | 27.8% |
| order_timing | 5 / 16 | 31.2% |
| attribute | 9 / 15 | 60.0% |
| camera_style | 5 / 10 | 50.0% |
| other | 2 / 7 | 28.6% |
| physics_motion | 3 / 7 | 42.9% |

## Cost

Per-clip wall time over 149 clips: median **102s**, mean 373.0s, p90 1086s, max 6119s. The mean exceeds the median because a minority of clips invoke dense per-frame specialists.

## Scope

- 149 of 150 evaluation-core clips; the shortfall is infrastructure (a reasoner endpoint dropped connections), not content.

- This set is the set the system was developed against, so it measures fit, not generalisation.

- Allegation hit rate is reported because recall alone rewards over-flagging.

- Code: https://github.com/WenzhuoXu/video_agentic_critic
