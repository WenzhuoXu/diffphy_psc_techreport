# EXP-029 — plan-based critic vs per-claim verification

All figures generated from raw per-clip rows by `analysis/techreport_results.py` and `analysis/techreport_failures.py`; recomputable from `techreport_results.json`. Bootstrap CIs use pre-registered seed 20260724, 10,000 resamples.

## Headline

Paired clip by clip on **149 clips**. The benchmark records 304 flaw entries, but 18 of them are duplicates (11 clips list the same span 2–3 times under different ids). The scoring loop credits one accusation to at most one flaw, so duplicates are uncatchable by construction; collapsing identical spans gives **286 distinct flaws**. Both figures are shown — the correction applies to every system equally and does not move the comparison.

| System | Flaws caught | Recall | As scored (with duplicates) | Calls/clip | Alleg. hit rate |
|---|---|---|---|---|---|
| Plan-based critic, all fixed specialists | 197 / 286 | **68.9%** | 197 / 304 (64.8%) | 13.0 | 40.1% |
| Per-claim verification, one claim at a time | 179 / 286 | **62.6%** | 179 / 304 (58.9%) | 12.1 | 40.2% |

**The plan-based critic catches 18 more flaws than per-claim verification** (95% CI [+8, +28], significant), at comparable cost per clip.

## Where the gain comes from

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

The largest gain is rendered text, where a recognition-head OCR engine reads glyphs off the pixels. A generative reader is unusable here: given the same crop it reported fluent English that is not on the object, while the OCR head returned the actual `"RRVLE"`, `"ToeFan"`.

## Why the misses are missed

Of **107 uncaught flaws**:

| Failure mode | n | Share |
|---|---|---|
| Checked and cleared — a claim covered it, the critic said the video was fine | 84 | 78.5% |
| Flagged by the critic, not credited by the scoring protocol | 20 | 18.7% |
| Never decomposed — no claim covers the flaw | 3 | 2.8% |

**No miss is caused by a specialist returning a wrong measurement.** 81 of 84 cleared flaws were decided by a bare yes/no judgement with no measurement attached. Only **14.1%** of executed checks are measurements.

Within those bare judgements, roughly half are not judge errors at all: the claim that was checked is a *weaker* assertion than the one the human flagged, because decomposition dropped the discriminating detail — a stated quantity, a spatial qualifier, or the second half of a compound sentence ("the two people lose balance, **arms flailing, and tumble overboard**" was checked as "Two people lose balance"). The judge answers that weaker claim correctly; the video is wrong about the part that was discarded. The remainder are genuine judge misses on the full assertion.

### Miss rate by category

| Category | Uncaught / total | Miss rate |
|---|---|---|
| action | 39 / 137 | 28.5% |
| count | 18 / 36 | 50.0% |
| object | 12 / 30 | 40.0% |
| text_ocr | 6 / 25 | 24.0% |
| spatial | 6 / 19 | 31.6% |
| order_timing | 6 / 17 | 35.3% |
| attribute | 10 / 16 | 62.5% |
| camera_style | 5 / 10 | 50.0% |
| other | 2 / 7 | 28.6% |
| physics_motion | 3 / 7 | 42.9% |

## Cost

Per-clip wall time over 149 clips: median **102s**, mean 373.0s, p90 1086s, max 6119s. The mean exceeds the median because a minority of clips invoke dense per-frame specialists.

## Scope

- 149 of 150 benchmark clips; the shortfall is infrastructure (a reasoner endpoint dropped connections), not content.

- This set has been used for development, so it measures fit, not generalisation.

- Allegation hit rate is reported because recall alone rewards over-flagging.

- Code: https://github.com/WenzhuoXu/video_agentic_critic
