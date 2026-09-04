# Phase82R+ — Performance-Maximizing Physical Tracking

This route is registered from the frozen Phase82P/Q0 state at the observed
start time. It first diagnoses why the Phase82P residual predicts `KEEP_Q0`
for every validation example, then applies one evidence-backed formulation:
class-balanced repair gate followed by reconnect ranking. Q0 proposals, base
score, physical lifecycle/evaluator denominator and sealed boundaries remain
frozen. TRAIN labels are used only to form targets and validation metadata;
category, text, semantic/physical IDs, future rows, held outcomes, DEV+, Q1,
public-new and sealed labels are never inference inputs.

The 10-hour window is `2026-09-04T02:17:53Z` through
`2026-09-04T12:17:53Z` (the start/deadline are recorded in
`outputs/iclr27_phase82r/audit/phase82r_registration.json`). GPUs 4–7 are
reserved, with one bounded worker per fold and at least 25% RAM free.

Registered order:

1. Read-only residual signal diagnostics (candidate recall, raw DINOv2
   separability, motion/geometry, history quality, candidate ambiguity).
2. Verify a per-video observed-step chronology and dormant-only candidate
   contract in a new implementation; retain canonical lineage remapping.
3. Train one balanced two-stage residual (repair-exists gate, then ranking)
   for 15 effective epochs with natural-distribution validation and safety
   selection. A checkpoint with zero predicted reconnect is labelled trivial,
   not best.
4. Extract native-event appearance once at measured throughput, replay the
   residual, run cheap physical proxy, and run formal TrackEval/strict O only
   if the proxy is non-inferior.
5. If residual authority is too conservative, automatically escalate to the
   registered selective Q0 overwrite, then full joint causal assignment. No
   threshold lottery or backbone/controller changes are allowed.

The selective overwrite uses one fixed TRAIN-only safety rule recorded in
`configs/iclr27_phase82r/selective_overwrite.json`: a learned reconnect is
accepted only when the gate probability is at least `0.9`; otherwise the exact
Q0 KEEP lineage is emitted. This is a single pre-registered conservative
fallback, not a held-event threshold sweep.

The frozen Q0 anchor is strict p16 both-reliable `25/76`; any learned route
must report physical proxy, TrackEval, strict O, retrieval and controller
metrics separately. The route may end only at the fixed deadline or a genuine
protocol/resource hard block; an individual model failure is not a task stop.
