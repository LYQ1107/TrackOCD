# TrackOCD Phase83 — Dual-Path preregistration

**Window.** Start `2026-09-04T07:43:07Z`; computational deadline
`2026-09-04T17:43:07Z` (ten hours, not a scientific stop wall).  The fixed
Luna/project identity and the repository `main` branch are unchanged.

## Questions

Phase83 tests two separable hypotheses under the frozen causal protocol:

1. Does the causal temporal-appearance-mean physical lineage (Phase82R) give
   the same raw DINOv2 correspondence evaluator better track-level signal
   than Q0?
2. With the proposal detector and boxes frozen, can a TRAIN-only causal
   support-assignment router move the frozen Phase75B observation result toward
   its proposal-pool ceiling without changing the denominator or evaluator?

Physical quality (`P`) and proposal/support observability (`O`) are measured
separately.  Neither result is an OCD Commit-CT result.  If either route has a
safe retrieval improvement, the registered next step is one corresponding
causal representation/controller experiment; no threshold, backbone or
StateMemory lottery is allowed.

## Immutable boundaries

- Q0/Phase75B rows, five-field row keys, candidate order, event denominator
  (76 positive + 76 negative), prefixes `{1,2,4,8,16}`, and reliable rule are
  read-only.  Frozen Phase75B O remains p16 `25/76` (source `49/76`, target
  `40/76`).
- Inference uses only current/past visual features, geometry, score, motion,
  age and history.  Category names/text, category or semantic labels, numeric
  physical IDs as features, future rows/tracks, held GT, DEV+, Q1, public-new
  and sealed labels are forbidden.  TRAIN GT can create loss/audit metadata
  only.
- Corrected DINOv2 cache is the only appearance anchor.  The old xywh/xyxy
  mis-cropped cache is not used.
- Phase82R Q0 and temporal-mean physical streams are frozen candidates.  The
  Phase75B evaluator is never edited; Phase83 O-support fields are versioned
  as `support_*_v1` in a separate artifact.

## Branch A: Physical → R → C

Use the exact Phase76/75D raw correspondence candidate universe and TRAIN-
disjoint folds.  Reconstruct track vectors from the same corrected DINOv2
rows, comparing Q0 physical lineage with the parameter-free causal temporal
mean.  Report query count/denominator, R@1, mAP, hard-negative gap, unsafe
flips and fold/category/video coverage.  Temporal mean is parameter-free and
cannot be selected using held events.

If raw R improves safely, a later route may train one representation and then
the unchanged causal controller.  If raw R is unchanged, run one diagnostic on
track pooling/source-support composition before choosing a single next route.

## Branch B: O-support

First produce a read-only callgraph and p16 positive failure taxonomy for
`assigned`, `row_iou`, and `track_temporal_iou`.  Recompute the frozen pool
upper bound from current artifacts (historical reference is approximately
source 72/76, target 64/76, both 61/76).  Event GT is post-hoc diagnosis only.

Then register exactly one small class-agnostic support-quality router using
non-event public TRAIN videos.  Inputs are candidate score/geometry,
corrected DINO appearance, temporal mean, motion consistency, track age,
history length, gap, proposal density and ambiguity.  Its TRAIN-only target
is the audited reliable-support condition; no event GT, category, text or ID
is an input.  Selection thresholds are frozen from TRAIN validation once and
the full event protocol is run once.  Report negative false-support activation
and coverage, not just positive recall.

## Resource/recovery contract

At most four GPUs (prefer idle GPUs 4–7), bounded workers, ≥25% RAM free,
one supervisor and one blocking wait for each long job.  Every unit writes an
atomic `.launched` then `.done`; checkpoints/cache files use temp+rename and
large artifacts live under `/data2/usr_for_deadline/trackocd_phase83` with
project symlinks.  An implementation failure gets evidence, the smallest fix,
compile/smoke/targeted regression and at most three repair cycles.  No broad
process kill and no external process may be touched.

## Gates

- `R83`: raw R improvement must satisfy the existing safety contract (no
  unsafe flips and at least three folds in the registered direction); this is
  a retrieval gate, not Commit-CT.
- `O83`: support-router evidence must be compared to a Q0 baseline under the
  new versioned evaluator, with negative safety and full 76+76 denominator.
- `C83` is authorized only after a safe R/O result and uses the unchanged
  causal controller.  Persistent Commit-CT, MOT safety and sealed status are
  always reported separately; missing stages are `NOT_RUN`, never zero.

