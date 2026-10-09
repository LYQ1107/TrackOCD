# PANDAS detection redundancy and ByteTrack recovery

Status: **COMPLETE**. Tracking-only NMS materially improves tracking on both
fixed TAO Validation subsets. It also loses Novel GT-covering candidates, so
no configuration passes all preregistered promotion gates. **No new full
988-video run was launched.** The completed frozen full-Val FG-0 result remains
the historical incumbent; subset results are not reported as full-Val metrics.

## Findings

Cross-prototype duplicates are real: 1,475,493 extra identical-geometry rows
are 39.38% of the fixed-20 cache. All 2,930,094 identical-box pairs have different
prototype IDs and exactly equal foreground scores. PANDAS repeats its
class-agnostic regression across prototype classes, flattens proposal × class
rows, and applies per-class NMS. ByteTrack treats surviving rows as separate
physical candidates. Proposal IDs are absent from NPZ, so the geometry/score
match is an observed proxy and not reconstructed raw lineage.

There are also many geometrically distinct proposals: 60.62% of cached rows
have distinct exact geometry. At IoU≥0.90, 69.38% of rows have an overlapping
partner. These partner statistics count both members and are not the fraction
removed by NMS or the number of distinct real objects.

The intervention demonstrates that the tracking interface is a major cause
of poor DetA/HOTA on the tested sets: independent-set HOTA rises from 0.122971
to 0.289395 and DetA from 0.040785 to 0.193343. However, the intervention combines
spatial NMS and a candidate cap. Its whole effect cannot be attributed only to
exact cross-prototype repetitions, and it does not prove they are the sole
bottleneck across all Val videos.

## Frozen inputs and actual source

- Ancestor: `411aed527d5c7999e4c88c10bed9be0ecd7b043d`.
- Branch: `codex/pandas-bytetrack-dedup-audit`.
- Audit, adapter and calibration commits: `0c39b60`, `140a930`, `2f33c7c`.
- COCO-half epoch-18 detector SHA256:
  `86ced2e8061722aae173905585fba204c0e085f16dd2b68dec9ccfbf2344a725`.
- Frozen K=500 prototype SHA256:
  `83cbe5f5e832414b4f0acd633964ade79e8d7e087923da3444ec26b9dba4a864`.
- PANDAS upstream: `9a6bafa728a14f3c7e936f1ee23d53110f362e85` plus the
  captured local foreground-score patch.
- Canonical TrackEval: `12c8791b303e0a0b50f753af204249e622d0281a`.

The previous actual execution entry points and their hashes are retained in
`scripts/baselines/pandas_bytetrack/frozen_runtime/` and
`source_manifest.json`. Their byte identity with the local original sources
was checked. The new adapter is
`src/baselines/pandas_bytetrack/tracking_postprocess.py`; new commands and
pinned dependencies are in the accompanying scripts README.

All 40 used source NPZ hashes rechecked successfully. Frozen checkpoint,
prototypes, detection-metrics JSON and incumbent full TrackEval JSON hashes
remain unchanged. Original outputs were neither overwritten nor copied into
new detection caches. No training, inference, discovery or TAO Test was run.

## Fixed sets and protocol

Calibration is the first 20 ascending video IDs:
`4, 20, 22, 23, 24, 25, 26, 30, 31, 35, 36, 37, 39, 45, 49, 52, 54, 55, 58, 59`.
It contains 22,322 complete-stream frames, 792 annotated images and 2,510 GT boxes
(2,014 Base, 222 Novel and 274 distractors).

Confirmation was fixed before metrics by the lowest SHA256 ranks of
`pandas-dedup-confirm-v1:<id>`, excluding calibration:
`91, 165, 308, 339, 387, 876, 1235, 1238, 1351, 1573, 1796, 1855, 2407, 2418, 2581, 2664, 2881, 2897, 2898, 2904`.
It contains 20,360 complete-stream frames, 723 annotated images and 2,290 GT boxes
(1,520 Base, 631 Novel and 139 distractors). No videos were chosen by performance.

ByteTrack is fixed at track/low/new = 0.50/0.10/0.60, match 0.80, buffer 30,
frame rate 30, MOT aspect filtering off, with `foreground_scores`. Original
FG-0 trajectories are reused. New NMS uses foreground ordering, ignores
prototype identity, resolves equal scores by original index, and preserves
selected semantic metadata. All frame/image IDs and empty frames are retained.
Source offsets are preserved verbatim; new candidate/track offsets reflect
suppressed rows. Track metadata joined by nearest IoU is only an audit aid,
not proposal lineage or an association input.

Tracking sees every frame; canonical TrackEval receives only TAO image-table
frames. Both runs use the exact same GT subset and pinned adapter. With
`SUBSET=all`, distractors are included as objects. Category placeholder 1 is
absent from negative/nonexhaustive flags in all 988 Val videos. Thus unmatched
predictions are ignored on GT-empty frames and count on GT-containing frames.
Some unmatched predictions may be unannotated real objects. The same limitation
applies to every old/new row. The reported MOTA uses native fraction values;
values below -1 are possible when unmatched predictions greatly exceed GT.

## Calibration results

| Variant | Candidates mean / median | Retained | Physical tracks | Median track length |
|---|---:|---:|---:|---:|
| Original-FG-0 | 167.87 / 162 | 100.00% | 51,125 | 10 |
| Dedup-A | 36.61 / 41 | 21.81% | 4,218 | 11 |
| Dedup-B | 42.17 / 50 | 25.12% | 5,186 | 12 |
| Dedup-C | 59.84 / 60 | 35.64% | 5,183 | 12 |

| Variant | HOTA | DetA | AssA | LocA | IDF1 | MOTA | IDSW | Frag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original-FG-0 | 0.101396 | 0.041197 | 0.253483 | 0.804879 | 0.045418 | -12.681673 | 821 | 153 |
| Dedup-A | 0.273374 | 0.204234 | 0.373223 | 0.814622 | 0.221288 | -1.230677 | 256 | 158 |
| Dedup-B | 0.232334 | 0.174916 | 0.315554 | 0.810703 | 0.171869 | -1.738645 | 394 | 160 |
| Dedup-C | 0.231965 | 0.174498 | 0.315322 | 0.810694 | 0.171464 | -1.748606 | 394 | 159 |

| Variant | All proposal recall | Base proposal recall | Novel proposal recall | Candidate GT matches / unmatched predictions | Track GT matches / unmatched predictions |
|---|---:|---:|---:|---:|---:|
| Original-FG-0 | 79.72% | 88.28% | 55.86% | 2,001 / 131,409 | 1,669 / 37,465 |
| Dedup-A | 77.37% | 86.79% | 48.20% | 1,939 / 27,145 | 1,608 / 5,018 |
| Dedup-B | 77.29% | 86.94% | 43.69% | 1,939 / 31,256 | 1,630 / 6,255 |
| Dedup-C | 79.04% | 87.98% | 52.25% | 1,983 / 45,396 | 1,629 / 6,281 |

Proposal recall counts GT boxes covered by any candidate at IoU≥0.5. The GT
match/unmatched columns use per-image spatial one-to-one matching at IoU≥0.5;
they are distinct from CLEAR's temporal matching counts (also saved in JSON).
"Physical tracks" and median length use every complete-stream output row,
not only the sparse annotated-frame export.

A is IoU0.5/max50, B IoU0.7/max50 and C IoU0.7/max100. The selection policy,
written before reading dedup metrics, requires absolute recall loss ≤0.02 for
All, Base and Novel. None is eligible: A loses 7.66 points of Novel recall, B
12.16 and C 3.60. As preset, C is locked for confirmation because it has the
least All recall loss. No new settings or ByteTrack thresholds were tested.

## Independent confirmation: locked C

| Variant | Candidates mean / median | Physical tracks | Median length | HOTA | DetA | AssA | LocA | IDF1 | MOTA | IDSW | Frag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original-FG-0 | 150.08 / 146 | 50,701 | 9 | 0.122971 | 0.040785 | 0.381371 | 0.815223 | 0.054478 | -13.029258 | 646 | 111 |
| Dedup-C | 49.91 / 50 | 4,334 | 11 | 0.289395 | 0.193343 | 0.443063 | 0.824355 | 0.233937 | -1.388210 | 260 | 112 |

C retains 33.25% of candidates and reduces physical tracks from 50,701 to
4,334. All proposal recall changes from 78.9956% to 78.5153%; Base from
88.2895% to 88.1579%; Novel from 58.6371% to 57.3693%. Independent-set Novel
loss is 1.27 percentage points (8/631), within the preset 2-point limit.

Candidate one-to-one GT matches/unmatched predictions change from
1,809/106,278 to 1,789/34,357. Track-output matches/unmatched predictions change
from 1,551/36,339 to 1,514/5,363. HOTA improves by 0.166425, DetA by 0.152557,
AssA by 0.061692 and IDF1 by 0.179459. Improvement is substantial while some
true GT coverage is still lost.

## Why the full run was declined

The preset full-run decision requires duplicate evidence, at least 20%
candidate reduction, safe All/Base/Novel recall on both sets, independent HOTA
and DetA gains of at least 0.005/0.002, and unchanged GT/evaluator. Six of the
seven Boolean gates pass. Calibration recall safety fails because C loses
8/222 Novel boxes (3.60 points). We retain this mixed result instead of relaxing
the gate after observing better HOTA.

An evaluator-only component audit found that calibration C's 17 coverage
losses (6 Base, 8 Novel, 3 distractors) all occur during NMS; max100 adds no
coverage loss. Three lost GT cases have an overlapping suppressor that covers
a different GT object, supporting an adjacent-object collision. Confirmation
C loses 11 GT boxes during NMS (2 Base, 8 Novel, 1 distractor), again with
three such collisions and no cap-only loss. A has 44 NMS losses plus 15 cap
losses; B has 17 NMS losses plus 44 cap losses. No new tracker variants were
needed for this decomposition.

The historical full-Val FG-0 metrics remain HOTA 0.110113, DetA 0.037554,
AssA 0.330246 and IDF1 0.048112. They cannot be directly substituted for the
subset FG-0 rows above, and there is no measured full-Val Dedup-C row.

## Separate localization, semantics and tracking findings

**Localization/proposals:** original native historical proposals cover 66.22%
of calibration Novel GT at IoU0.5. The current ≥0.01 frozen cache covers only
55.86%, and proposals meeting the fixed birth score 0.6 cover 20.27%. The
original cache is a separate frozen inference run; these comparisons are
descriptive, not an isolated measurement of the storage floor. NMS cannot
recover missing Novel boxes, and spatial suppression can lose covered ones.

**Tracking interface:** repeated prototype-conditioned rows and overlapping
proposals greatly inflate physical candidates and births. The controlled
intervention sharply reduces track count and raises DetA/HOTA, confirming an
important interface bottleneck independent of the unchanged model/discovery.
The remaining proposal limitations and collisions prevent claiming that the
whole problem is solved.

**Semantic discovery:** frozen Novel AP remains 0.002249334945614304; Base AP
0.09461081464386155 and All AP 0.020650855805035215 are unchanged. NMS changes
only tracking input. In an evaluator-only, GT-covered, highest-semantic-score
candidate diagnostic, correct frozen Hungarian-mapped Novel labels occur on
22/124 calibration cases (17.74%) and 121/370 confirmation cases (32.70%).
Purity is 100%/92.97% on those restricted covered subsets, but high purity with
many small prototypes does not guarantee a one-to-one semantic mapping or high
AP. The prototype counts include base prototypes. Mapping was fitted on TAO
Val, making these conditional figures optimistic; they are not independent
New ACC, a full clustering-quality estimate or an AP causal decomposition.
Proposal gaps plus weak mapped assignment/ranking are consistent with low
Novel AP; this tracking adapter does not repair Novel Discovery.

## Operating point and next research

No dedup setting is promoted to a full-Val replacement under the declared
recall protection criterion. Keep frozen FG-0 as the historical measured
PANDAS full-Val reference. C (IoU0.7/max100, fixed FG-0 ByteTrack) is the
reproducible tracking-only diagnostic operating point with independent evidence
of benefit, and must be labeled as TAO Val development with a failed calibration
Novel recall gate. A has the best calibration HOTA but higher recall loss and
was not independently confirmed.

Further PANDAS threshold search, retraining or rediscovery is not justified
within this completed diagnostic. Return to the already selected fixed physical
tracker and common features, then compare Nearest / DP-Means / PHE / TrackOCD.
The next performance claims should come from persistent category discovery.

## Validation and delivery

Seven targeted tests passed, including exact CPU/GPU agreement on the tested
geometry, score tie determinism, metadata preservation, no GT/text inputs,
unchanged frame identities/source offsets and exactly one update per empty
frame. Six canonical subset TrackEval runs returned PASS. Original checkpoint,
prototype, NPZ, detector AP and full FG-0 artifacts stayed read-only. Added
output is approximately 61 MiB against a 10 GiB soft budget; all tracking and
audits were CPU jobs, with a small GPU unit test only. No external process was
modified or stopped.

Commands and pinned source are in `scripts/baselines/pandas_bytetrack/README.md`.
Detailed evidence is in `outputs/baselines/pandas_dedup/` (audit, calibration,
confirmation, locked variant, decision policy and suppression loss audit).
Large outputs remain outside Git; only small JSON evidence, source and reports
are committed. The final machine-readable report is
`PANDAS_BYTETRACK_DEDUP_RESULT.json`. Final branch/remote HEAD equality is
recorded after the delivery push in the local `git_delivery.json` evidence.
