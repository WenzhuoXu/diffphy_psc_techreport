# EXP-029 — plan-based critic vs per-claim verification: expanded results

Regenerated 2026-07-26 from the current rows (an earlier snapshot of this file read 135 clips / 286 flaws). Generated from raw per-clip rows by `analysis/techreport_results.py` and `analysis/techreport_failures.py`. Every figure here is recomputable from `analysis/techreport_results.json`; nothing is hand-typed. Bootstrap CIs use the pre-registered seed 20260724 with 10,000 resamples.

## Headline: paired comparison

Scored on **137 clips** carrying **289 human-labelled flaws**, paired clip by clip, one matching protocol.

| System | Flaws caught | Recall | Calls/clip | Allegation hit rate |
|---|---|---|---|---|
| Plan-based critic, all fixed specialists | 190 / 289 | 65.7% | 13.2 | 40.9% |
| Per-claim verification, one claim at a time | 173 / 289 | 59.9% | 12.3 | 40.3% |

**The plan-based critic catches 17 more flaws than per-claim verification** (95% CI [+7, +27], significant), at comparable cost per clip.

### Where the gain comes from

| Flaw category | n | Plan-based critic | Per-claim |
|---|---|---|---|
| action | 130 | 94 (72%) | 86 (66%) |
| count | 35 | 17 (49%) | 15 (43%) |
| object | 29 | 18 (62%) | 18 (62%) |
| text_ocr | 21 | 16 (76%) | 12 (57%) |
| spatial | 17 | 12 (71%) | 10 (59%) |
| order_timing | 16 | 10 (62%) | 10 (62%) |
| attribute | 15 | 6 (40%) | 5 (33%) |
| camera_style | 10 | 5 (50%) | 5 (50%) |
| other | 7 | 5 (71%) | 4 (57%) |
| physics_motion | 6 | 4 (67%) | 5 (83%) |

The largest single gain is **rendered text**, where a recognition-head OCR specialist reads the glyphs off the pixels instead of asking a generative model what the sign says. A generative reader's language prior repairs garbled text and hides the very defect being hunted; on one clip the served VLM reported fluent English that is not on the object, while the OCR head returned the actual `"RRVLE"`, `"ToeFan"`.

## Why the misses are missed

Of **99 uncaught flaws**:

| Failure mode | Count | Share |
|---|---|---|
| Checked and cleared — a claim covered it and the critic said the video was fine | 77 | 77.8% |
| Flagged by the critic, but the matching protocol did not link it to the human flaw | 19 | 19.2% |
| Never decomposed — no claim in the plan covers the flaw | 3 | 3.0% |

**74 of 77** cleared-but-wrong flaws were decided by a yes/no judgement with **no measurement attached** — not by a specialist returning a wrong number. Decomposition is not the bottleneck (3 of 99); measurement *coverage* is.

Only **14.6%** of executed checks are measurements; the rest are occurrence/existence judgements. Check-kind census: `{"event_occurs": 1315, "existence": 228, "audio_event": 173, "spatial": 22, "trajectory": 19, "text_content": 17, "count": 16, "temporal": 16}`.

### Miss rate by category

| Category | Uncaught / total | Miss rate |
|---|---|---|
| action | 36 / 130 | 27.7% |
| count | 18 / 36 | 50.0% |
| object | 11 / 29 | 37.9% |
| text_ocr | 5 / 22 | 22.7% |
| spatial | 5 / 17 | 29.4% |
| order_timing | 6 / 17 | 35.3% |
| attribute | 9 / 15 | 60.0% |
| camera_style | 5 / 10 | 50.0% |
| other | 2 / 7 | 28.6% |
| physics_motion | 2 / 6 | 33.3% |

## Cost

Per-clip wall time over 137 clips: median **95s**, mean 391.8s, 90th percentile 1371s, max 6119s. The mean far exceeds the median because a minority of clips invoke dense per-frame specialists; those set the wall-clock, not the typical clip.

## Scope and honesty

- Paired set is **137 of 150** benchmark clips. The shortfall is infrastructure, not content: a reasoner endpoint dropped connections mid-pass, leaving some clips without a decomposed claim set. Recovery is in progress.

- The four-way set (adding the two earlier systems) is 107 clips / 229 flaws; the earlier-planner arm was not extended, so it is reported as a secondary comparison only.

- This set has been used for development, so it measures fit, not generalisation.

- Allegation hit rate is reported because recall alone rewards over-flagging; the critic makes many accusations per clip and only a minority match a human label.
