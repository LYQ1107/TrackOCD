# TrackOCD Phase86 preregistration

Phase86 is a bounded diagnostic-and-research window. The diagnostic OCD replay
is frozen, read-only, and marked `DIAGNOSTIC_ONLY_DO_NOT_SELECT`; it cannot
choose checkpoints, thresholds, architectures, candidate counts, or routes.

## Frozen boundary

- Phase85/Q0/temporal/support artifacts are read-only.
- The denominator is 76 positive + 76 negative events at prefixes
  `{1,2,4,8,16}`. The 984-query retrieval family, candidate order, and
  same-video exclusion remain unchanged.
- DEV+, Q1, public-new, sealed labels, future rows/tracks, category/text,
  semantic IDs, and physical IDs as model features are forbidden.
- TRAIN labels may be used only for TRAIN-fold supervision or post-hoc scoring.
- At most four idle GPUs may be used. Large outputs live under `/data2` and
  are exposed through the Phase86 project symlink.

## Frozen controller rule

The controller is selected by chronology and prior formal use, never by the
Phase86 76+76 outcome. The last byte-identifiable controller formally frozen
and used in the legal Commit-CT protocol is the Phase19R RC-MS-OCD controller
(`src/iclr27_phase19r/models/controller.py`) with its shared causal
`StateMemory` (`src/iclr27_phase19r/runtime/state.py`) and the fold-local
`fold{0..3}_best_internal.pt` checkpoints. Phase56's new unified controller
failed its registered Gate C and is retained as historical negative evidence,
not substituted as a Phase86 baseline.

## Pre-registered upstream order

1. D0--D3 frozen diagnostic OCD streams, if each has a legal adapter.
2. U1 OOF selective intervention gate over the frozen Phase85 raw and bounded
   reranker experts. Utility is fixed (`HELP=+1`, `HARM=-4`) and learned from
   TRAIN held-out predictions only. Formal replay is allowed only when
   `net_rescue > 0` and `harm <= rescue` on at least two validation folds.
3. If U1 TRAIN evidence fails, audit verified 2025/2026 correspondence methods
   and run the single registered U2 source-query relation encoder. U2 retains
   the raw anchor and is selected on TRAIN validation only.
4. In parallel, run the parameter-free Root-of-Fragments (RF) diagnostic:
   symmetric fragment matching with fixed `0.05*tanh` raw-anchored residual.
   RF is not trained and cannot modify physical IDs.
5. Alignment, formal OCD, and D4 are downstream only when their registered
   gates are satisfied. No diagnostic outcome is fed into training.

## Stop/repair rule

Keep failed markers and artifacts. For an implementation failure, perform only
the smallest repair, compile/import smoke, and one targeted regression before
resuming. At most three repairs address the same root cause and route. No
threshold, memory, controller, backbone, denominator, seed, or evaluator
lottery is authorized.
