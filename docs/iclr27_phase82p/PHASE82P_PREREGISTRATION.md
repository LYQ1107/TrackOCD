# TrackOCD Phase82P+ preregistration

## Objective

Test whether a conservative, Q0-anchored repair of birth/fragment transitions
can recover missed causal continuations without degrading the frozen physical
MOT stream.  The route is deliberately residual: every non-birth Q0 row is
unchanged, and a birth may only reconnect to a causal dormant Q0 fragment from
the same video within the preceding 16 frames.

## Frozen contract and boundaries

The OVTR/Q0 proposal boxes, base score, chronological rows, parent assignment,
and non-birth continuation decisions are frozen.  Physical IDs are runtime
bookkeeping only.  Inference tensors contain visual/geometry/motion/history and
quality features, never category names/text, category or semantic IDs,
physical-ID values, future rows, held labels, DEV+, Q1, public-new labels or
GT.  TRAIN GT is used only offline to label a reconnect target.  Event videos
are excluded from residual fitting and validation; validation is video-disjoint.

## Registered route

At each Q0 birth, construct the actual causal candidate set of dormant/lost Q0
tracks whose last observation is at most 16 frames old.  Rank/prune at most 16
with a fixed combination of causal motion compatibility, visual similarity and
recency.  Each candidate contributes eight past observations (`K=8`) containing
normalized box geometry, base score, frame gap, velocity, age, association
quality and an appearance descriptor.  A two-layer, four-head Transformer with
256 hidden units encodes the history; a proposal/pair head scores `KEEP_Q0` and
each candidate reconnect action.  Invalid or empty support is an exact
`KEEP_Q0` fallback.  If reconnect wins, the new birth lineage is remapped to
the old Q0 ID from that frame onward with one-to-one conflict checks.

The listwise loss is cross entropy over the actual `KEEP_Q0 + candidates`
distribution.  A false-reconnect penalty has weight 2 and a missed-repair
penalty weight 1.  No threshold sweep is registered.  Checkpoint selection is
lexicographic: minimum false reconnect, maximum repair precision, maximum
repaired fragments, then recall.

## Required audits and gates

Before training, run the exact Phase75B strict O wrapper on native Q0 and
reproduce 25/76 both-reliable events at prefix 16.  Build a per-video manifest
and verify history reset, true sequence length K=8, no event-video leakage and
no forbidden tensor fields.  Compare cheap physical proxies (tracks, switches,
fragmentation, merges, duplicate births, correct/harmful reconnect) against Q0.
TrackEval is run only after a route is non-worse on these proxies.  Strict O is
reported with its fixed `assigned == 1 and transformed IoU >= 0.5` contract.

The residual route is informative if it improves strict p16 both-reliable events
(35/76 target, 40/76 strong, 50/76 major) while preserving Q0 physical safety;
these are research targets, not replacements for the final MOT+OCD gates.  If
residual repair is safe but small, continue to the pre-registered selective
overwrite route.  If unsafe, perform an evidence-based repair and then continue
to selective overwrite or full causal association before the ten-hour deadline.

## Resources and reproducibility

Use at most four bounded workers, one each on GPUs 4–7, with at least 25% RAM
free.  Every unit writes an atomic `.launched` marker at spawn and `.done` at
completion; checkpoints are resumable every 1000 updates.  One supervisor and
one blocking wait are used for long jobs.  All large outputs live under the
Phase82P `/data2` symlink; old phase artifacts remain read-only.  Actual start,
deadline, hashes, resources and every repair event are recorded in
`outputs/iclr27_phase82p/audit/` and `research_log.md`.
