# R1 — the single controlled GT representation correction

Preregister before fitting; original fits/results from `ba44bb5` and policy-only
diagnosis `130ac95` remain unchanged. Primary frontend stays blocked and this
is not a formal M8/main result. The single root-cause correction budget is
consumed when this run starts; no second architecture/loss/sampling/gate retry.

Evidence: policy_train-only prototype-score AUROC worsens after adaptation,
so threshold shift alone is not the entire observed problem. Investigate
preserving frozen visual open-set geometry during the tiny four-class fit.
This is a falsifiable correction hypothesis, not a uniquely proved cause.

Only change: add weight-5 teacher-geometry Gram MSE to both A1 and A2.
For the same augmented observed-prefix fit batch, L2-normalize frozen DINO
prefix means and detach their cosine Gram; align the learned evidence cosine
Gram with it. This uses no new class supervision, prototype, policy, heldout,
Val or Test sample in the optimizer. It is a geometry regularizer, not a new
detector/backbone, oracle semantic memory or global GT mapping.

Everything else unchanged: exact 12 fit tracks / 192 observations, architecture
and parameter count, seeds 1027–1029, 120 steps, optimizer/LR, shared corruption,
auxiliary reliability loss, four prototype tracks, .65/.55 gates, stream rows,
four orders and p1/p2/p4/p8/p16. Same 600-s / 64-MiB / one-worker resource caps.
Pinned parent config hash prevents covert protocol drift. Save new checkpoints
and results in `representation_r1`, never overwrite `representation`.

Freeze all six final R1 fits before evaluation. Retain every seed/order/prefix,
fixed denominator and error type. Compare both R1-versus-original and paired
A2-versus-A1; no checkpoint/prefix cherry-picking. Calibration and module/aux
loss confounds remain limitations, not silently claimed resolved. Report a
negative result honestly and do not authorize M11 from this tiny experiment.

## Reproduction

Use `train_gt_representation_pilot.py --config
configs/trackocd_core/gt_representation_correction_r1.json`, then the same
config with `evaluate_gt_representation_pilot.py`. Raw features/checkpoints
stay private. Both preserve previous completed artifacts.

## Actual result — correction cap exhausted, temporal claim not supported

Commit `63a3ad0` was pushed and independently remote-verified before fitting.
Six R1 fits completed in 13.55 s, GPU peak 28.73 MiB / host ~1.06 GiB;
new private checkpoints 12,822,732 B. Frozen CPU evaluation took 9.41 s,
280 replays / 6,720 decisions, ~677.14-MiB peak. A0 still reproduces the
original B1 exactly. Full per-seed/order/prefix JSON/CSV and private sealed
prediction hashes are retained separately from the original experiment.

Heldout selection p16 percentages (four-order means):

| Model | Seed | Old ACC | New ACC | H | Correct Commit-CT | Wrong Known |
|---|---:|---:|---:|---:|---:|---:|
| A1 R1 adapter/mean | 1027 | 83.33 | 0.00 | 0.00 | 0.00 | 100.00 |
| A2 R1 adapter/evidence | 1027 | 83.33 | 0.00 | 0.00 | 0.00 | 100.00 |
| A1 R1 adapter/mean | 1028 | 70.83 | 10.42 | 17.85 | 0.00 | 75.00 |
| A2 R1 adapter/evidence | 1028 | 70.83 | 10.42 | 17.85 | 0.00 | 75.00 |
| A1 R1 adapter/mean | 1029 | 83.33 | 16.67 | 26.81 | 12.50 | 81.25 |
| A2 R1 adapter/evidence | 1029 | 83.33 | 16.67 | 26.81 | 12.50 | 81.25 |

A1/A2 both average Old 79.17%, New 9.03%, H 14.89%, Correct CT 4.17%,
wrong Known 85.42%. Some discovery improves over the original all-Known
rejection failure, but two seeds still have zero correct p16 reuse. A2 has
no p16 gain over A1 in any seed; at p1 only seed 1027 gains 9.375 CT points,
at p8 the same seed loses 6.25 points, and p2/p4/p16 paired CT deltas are
all zero. This is not stable independent temporal evidence benefit.

Decision: retain simpler A1 as a **candidate**, not a scientific PASS. Do not
promote A2, hide failures, pick a favorable prefix or launch R2. The single
controlled correction is consumed. Model/aux-loss and calibration confounds,
tiny four-class sampling, GT-only geometry and unqualified primary frontend
remain limitations. No formal M4/M5/M8/M9 completion or M11 authorization.

Next in-scope recovery work: lawful primary-frontend provenance/assets and
protocol completion. A broader GT-only decision-learning branch is distinct
from the explicitly allowed representation-feasibility exception and needs
scope clarification before using it to bypass the strict primary-stage order.
