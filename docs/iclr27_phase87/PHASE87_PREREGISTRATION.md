# TrackOCD Phase87 — Causal Persistent OCD Controller Redesign

**Window:** 2026-09-06 17:53:10 UTC → 2026-09-07 03:53:10 UTC  
**Code head at registration:** `d6c193efec3accdfd01e52ec526000c452182b9e`  
**Status:** registered before training; Phase86 is read-only.

## Hypothesis

The Phase86 failures are partly an interface failure: a frozen score selector is
being forced into an incompatible threshold-chain controller. A causal,
transactional semantic state with explicit EXISTING/NEW/DEFER/RESET actions can
learn persistent evidence without changing the physical MOT stream. The test is
not a claim that retrieval will translate to Commit-CT; diagnostic OCD is run
separately and remains the decisive downstream check.

## Fixed data and leakage boundary

- TRAIN supervision only, four existing video/category-disjoint folds.
- All 91 videos appearing in the Phase19R/Phase30 event observability manifest
  are excluded from controller fitting and validation event construction.
- A source track must be in a strictly earlier video than its target; source
  and target physical tracks are distinct. Category labels are loss metadata
  only, never model tensors.
- No DEV+, Q1, public-new, sealed labels, future rows/tracks, category text,
  semantic IDs or physical IDs enter inference tensors.
- Prefixes are causal and monotone. The original 76 positive and 76 negative
  event denominator is preserved for a diagnostic-only replay.
- No threshold sweep, candidate deletion, denominator/seed/evaluator change,
  or checkpoint selection from held events is permitted.

## Architecture contract

`CausalTrackEncoder` consumes raw 768-D visual features and 15 causal geometry
fields with a one-layer GRU (hidden 256) and a bounded `0.05*tanh` residual;
semantic output stays 768-D. `StateMemoryV2` stores at most 16 states and four
prototypes per state. A candidate is eligible only when both
`birth_video != current_video` **and** `birth_track != current_track`.
`TargetSession` accumulates evidence causally and commits only inside the
current track. Global memory mutates transactionally at track end: EXISTING
updates a selected state, NEW creates one, and DEFER/RESET does not mutate
global memory. The action space is state slots plus NEW/DEFER/RESET; there is
no threshold-chain risk decoder.

## Training contract

The single C0 route uses the shared train/eval rollout, AdamW (`lr=2e-4`,
`weight_decay=1e-4`, gradient clip 5), seed base 87000, 20,000 updates per
fold, checkpoints every 2,000 updates, and BF16 autocast when supported. The
loss is the fixed sum of action cross-entropy, state relation, false-merge
risk, new-vs-existing margin, and commit/defer margin. Support features are
zeros in C0; C1 support integration is allowed only after a TRAIN C0
improvement gate. Reset supervision is retained in the action space but is not
invented when no legal reset target exists.

Execution order is contract checks → TRAIN baseline → 100-step smoke → fold0
500-step targeted → four-fold C0 formal. Each unit has an atomic `.launched` and
`.done` marker and writes checkpoints atomically under `/data2`. The formal
gate is historical and fixed: Commit-CT ≥15, category coverage ≥5, video
coverage ≥8, existing precision ≥0.70, negative false merge ≤0.15, known
micro ≥0.206, and known macro ≥0.139. These gates are not relaxed because of
an intermediate result. If C0 fails, one registered safety-weight repair may
be attempted (false merge, unresolved, duplicate-birth, or reset weighting only
when its evidence condition is met); then C1 is conditional on improvement.

## Stopping and provenance

Three repair cycles for the same root cause end that route only. The overall
window remains open until the registered route is complete or the final report
records the evidence-based decision. Public/sealed evaluation is not authorized
in Phase87. All code and the final report are pushed to the public TrackOCD
repository before handoff.
