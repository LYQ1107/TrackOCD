# PANDAS detection redundancy audit

Status: completed read-only fixed-20 audit; tracking intervention is still to be evaluated.

The calibration set is the first 20 ascending TAO Validation video IDs:
`4, 20, 22, 23, 24, 25, 26, 30, 31, 35, 36, 37, 39, 45, 49, 52, 54, 55, 58, 59`. The disjoint confirmation set was
fixed before metrics using SHA256(`pandas-dedup-confirm-v1:<id>`):
`91, 165, 308, 339, 387, 876, 1235, 1238, 1351, 1573, 1796, 1855, 2407, 2418, 2581, 2664, 2881, 2897, 2898, 2904`.

## Actual execution source

PANDAS upstream commit `9a6bafa728a14f3c7e936f1ee23d53110f362e85` plus the
recorded foreground-export patch is the execution source. Its predictor repeats
a class-agnostic box regression for every prototype; RoIHeads flattens each
proposal × class into detections and performs `batched_nms` with prototype
labels. Surviving class-conditioned rows select foreground confidence by the
same internal `box_id`. ByteTrack receives all such rows as independent
physical candidates. There is no category condition in ByteTrack association.

The NPZ fields are `frame_index, image_id, frame_offsets, boxes, scores,
foreground_scores, prototype_id`. They support this audit and tracking-only
postprocessing without inference. Raw proposal IDs are not saved, so origin
matching from identical geometry and confidence is an observable proxy, not
recovered proposal lineage.

The actual preceding entry points are captured in
`scripts/baselines/pandas_bytetrack/frozen_runtime/`; their hashes and pinned
upstream revisions are in `source_manifest.json`. The local model patch is
also retained. Neither mainline code nor the legacy result artifacts changed.

## Detection counts and overlap

All 22,322 full-stream frames were read one video at a time.
The current frozen cache has 3,747,198 rows in this set.
The historic smoke count (3,736,349) is not substituted for the current complete
cache evidence (3,747,198). The original native historical cache has 300
rows/frame; it is a separate frozen inference run, so differences against it
are descriptive comparisons, not isolated numerical effects of the floor.

| Candidates | Mean/frame | Median/frame |
|---|---:|---:|
| Native historical cache | 300.0000 | 300 |
| foreground ≥0.01 | 167.8702 | 162 |
| foreground ≥0.1 | 93.0730 | 76 |
| foreground ≥0.5 | 54.6037 | 42 |
| foreground ≥0.6 | 47.1934 | 36 |
| Exact unique geometry | 101.7698 | 95 |
| Greedy IoU 0.99 geometry estimate | 101.7399 | 95 |

Exact repetitions beyond the first comprise 39.38%
of cached rows (1,475,493 rows).
All 2,930,094 identical-box pairs have different prototype IDs
and exactly equal foreground confidence.

| IoU threshold | Candidates with overlapping partner | Cross-prototype pair fraction | Equal-foreground pair fraction |
|---|---:|---:|---:|
| ≥0.9 | 69.38% | 100.00% | 54.87% |
| ≥0.95 | 63.94% | 100.00% | 83.57% |
| ≥0.99 | 60.81% | 100.00% | 99.92% |

"Candidates with partner" counts a candidate once if any other row reaches
the threshold; it is not the fraction removed by NMS and can count both members
of a pair. Exact unique geometry is not a count of unique real objects.

## Evaluator-only proposal recall at IoU 0.5

| Candidate universe | All | Base | Novel | Distractor |
|---|---:|---:|---:|---:|
| original_pre_floor | 83.55% | 90.32% | 66.22% | 47.81% |
| 0.01 | 79.72% | 88.28% | 55.86% | 36.13% |
| 0.1 | 73.47% | 84.01% | 34.23% | 27.74% |
| 0.5 | 65.98% | 77.56% | 20.72% | 17.52% |
| 0.6 | 64.22% | 75.77% | 20.27% | 14.96% |

Denominators are 2,510 GT boxes: 2,014 Base, 222 Novel, 274 distractors.
Recall means any candidate covers a GT box, not correct anonymous category
prediction. GT does not enter tracking postprocessing or parameterization.

## Interpretation and evaluator policy

Cross-prototype duplicates are real and numerous, and the source explains their
mechanism. However, **60.62% of rows have distinct exact geometry**; the cache
also contains many spatially different proposals. The overlap statistics alone
do not establish that duplicate detections are the primary tracking bottleneck,
or that every near-overlap represents the same real object. The three fixed
NMS interventions and disjoint confirmation set will test that hypothesis.

Novel localization is already constrained: the current ≥0.01 cache covers
55.86% of the subset's Novel boxes,
falling to 20.27% at the fixed birth threshold.
Removing duplicates cannot recover absent proposals or repair anonymous
semantic assignment. Frozen Novel AP remains 0.002249334945614304.

Pinned TAO-OW (`12c8791b303e0a0b50f753af204249e622d0281a`) exports only
36,375 TAO image-table frames, after full-stream tracking. `SUBSET=all`
collapses all GT, including distractors, into object. Category placeholder 1
is absent from negative/nonexhaustive flags in all 988 videos. Consequently
GT-empty frames' unmatched detections are ignored and unmatched predictions on
GT-containing frames count toward DetA/MOTA. Some may be unannotated real
objects; unmatched predictions are not all verified false objects. Every
old/new comparison uses this exact policy, GT and evaluator.

Evidence and per-file hashes: `outputs/baselines/pandas_dedup/audit.json`.
No input NPZ was modified; pre/post checksums agree. No inference, training,
discovery or TAO Test access occurred. The audit used CPU and stayed within the
10 GiB additional-output budget.

## Intervention outcome

The three fixed NMS/candidate-cap interventions and disjoint confirmation have
now completed. C (IoU 0.7, max 100) increased HOTA from 0.101396 to 0.231965
on calibration and from 0.122971 to 0.289395 on confirmation. This confirms a
major tracking-interface bottleneck. Exact cross-prototype duplication is one
identified mechanism, but the intervention also suppresses distinct overlapping
proposals and caps candidates; these effects must not all be credited to exact
duplicates. C lost 8 previously covered Novel GT boxes in each subset due to
NMS, including adjacent-object collisions. Calibration Novel recall loss was
3.60 percentage points, exceeding the preregistered 2-point protection limit.
The full-Val promotion gate therefore failed and no new 988-video run was
launched. The final report separates this tracking benefit from proposal and
semantic limitations.
