# Actual frozen Train-heldout policy and unseen-category findings

Status: completed controlled scientific negative result,not predicted-track
evaluation/final goal completion.18real fits,1800sealed evaluation cases:
3seeds x10methods x4orders x5caps x3development coverage-target points.
207Known/16pseudo-Novel fixed targets;CT opportunity denominators9/7/7/9.
Actual evaluation256.8800s oneCPU,peak847216640B. No extra fit/threshold trials.

## p16 development target1.0 operating points

All values are percentages,first average four orders per training seed,then
average three seeds. H is the mean of actual H values,not H recomputed from
averaged Old/New;different seeds can recognize disjoint semantic roles.

| Method | Old | New | Mean H | Correct CT | False Merge | Novel reuse commitment coverage |
|---|---:|---:|---:|---:|---:|---:|
| A1 D1 MLP | 50.89 | 7.81 | 11.82 | 0 | 0 | 100 |
| A1 D2 risk | 51.53 | 20.83 | 0 | 0 | 30.95 | 97.62 |
| A2 D1 without risk | 62.00 | 5.21 | 6.92 | 0 | 0 | 100 |
| A2 D2 full | 51.21 | 20.83 | 0 | 0 | 30.03 | 96.69 |
| A2 D2 without WAIT | 51.21 | 20.83 | 0 | 0 | 33.33 | 100 |
| A2 D2 without memory | 51.21 | 2.08 | 0 | 0 | 0 | 72.75 |
| A2 D2 reset each video | 51.21 | 12.50 | 0 | 0 | 0 | 80.16 |

Every one of the1800cases has0correct pure cross-video reuse. D2 does not
support persistent-decision PASS. Its lower wrongKnown on some seeds is
accompanied by wrong/contaminated anonymous merging,not safe category reuse.
Full's mean54.42memory contamination writes atp16,target1.0 are real;reset
and no-memory lower merge partly by removing reuse or lowering coverage.
WAIT can decrease falseMerge .3333→.3003 but also changes effective coverage
1→.9669,total target commitment1→.9152. This is not a matched-risk advantage.

540paired risk/temporal comparisons include377within the preregistered BOTH
coverage-gap<=.05 criterion;163are explicitly NOT_COMPARABLE_AT_MATCHED_COVERAGE.
All finite operating choices and actual gaps remain in the JSON. The nominal
targets .5/.75/1 are not claimed to be realized on every seed. No interpolated
favorable point or infinite WAIT success. WrongKnown,mixed state,fragmentation,
WAIT and conditional bootstrap diagnostics are retained for all cases.

## Unseen-category retrieval,not fitting-Known probe inflation

All16pseudo-Novel queries have cross-video positives;gallery also includes
Known negatives. Four actual heldout pseudo classes are sparse and imbalanced.

| Prefix | Raw pseudo-Novel macro Rank1 | Selected A1 mean | A2 mean |
|---|---:|---:|---:|
| p1 | .2000 | .1583 | .1583 |
| p2 | .3000 | .1583 | .1583 |
| p4 | .4500 | .1833 | .1750 |
| p8 | .4500 | .2250 | .2250 |
| p16 | .3500 | .3083 | .3083 |

Overall category Rank1 gains reported earlier mainly reflect Known probes.
A1 cannot claim unseen Rank1 improvement at any cap;A2 has no stable independent
advantage and does not beat its matched static control. Some higher-K recall
and AUROC improve,so describe partial gains rather than claiming all geometry
collapsed or all metrics worsened. Nonetheless correct persistent discovery
and top1unseen retrieval are not supported;TEMPORAL_EVIDENCE_NOT_SUPPORTED.

## Evidence-backed failure interpretation

Observed:learned embeddings concentrate (effective rank54.18→8.59),Known
recognition improves,unseen top1retrieval worsens,nearest anonymous memory
merges more,and learned policy commits mostly wrongKnown or polluted states.
Only the best Known ID/anonymous token is selectable;loss targets sometimes
must WAIT when the correct candidate is not represented.20epochs/80policy
updates and15pseudo policy tracks/four classes offer limited transferable risk
experience. These are plausible limitations,not separately randomized causal
proofs of why each failure happened. No post-heldout capacity/loss/budget rescue.

RawFrameVote remains a stronger controlled persistent baseline;PHE remains
INCOMPARABLE. Old R1 is preserved and this independent enlarged protocol is
not R2. Frozen negative models still proceed to the requested limited MASA
comparison,all unknown/unmatched tracks and full GT denominators retained.
