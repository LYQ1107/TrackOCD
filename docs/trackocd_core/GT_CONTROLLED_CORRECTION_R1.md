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
