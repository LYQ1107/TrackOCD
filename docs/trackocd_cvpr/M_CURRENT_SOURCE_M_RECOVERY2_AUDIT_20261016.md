# Current-source M recovery-2 matrix audit — 2026-10-16

## Scope and decision

This is a read-only, machine-recomputed audit of the nine explicitly
registered recovery-2 M development receipts under
`outputs/trackocd_cvpr/cvpr4/M_GT_causal_dev_current_source_wall_recovery2_*`.
It does not open new dataset files, train, select a checkpoint, read TAO Val or
Test labels, or change any scientific acceptance flag.

The audit script is
`scripts/trackocd_cvpr/audit_current_source_M_recovery2.py`, and its ignored
machine output is
`outputs/trackocd_cvpr/cvpr4/M_current_source_recovery2_matrix_audit_v1/result.json`.
It returned:

`PASS_CURRENT_SOURCE_M_RECOVERY2_MATRIX_AUDIT`

All 9 expected cases were present and valid: `relative_quarantine`,
`relative_weighted`, and `pooled_weighted`, each at seeds `1027`, `1028`, and
`1029`. Every receipt contains order probes `1027/1028/1029`.

## Boundary checks

All 9/9 cases and 27/27 order probes passed the following checks:

- status is `MEASURED_CAUSAL_GT_M_DEVELOPMENT_NOT_ACTIVE_PURITY_SELF_EXCLUSION_PRED_OR_GOAL_ACCEPTANCE`;
- `optimizer_steps=0` and model state is unchanged;
- the full physical-ID reverse-stream probe is true;
- the future-suffix probe mutates 39 observations and preserves the prior-event
  chosen versions and numeric ledger;
- every order is explicitly `GT_oracle_control_not_Pred`;
- no Val/Test/new-nonKnown semantic values were read;
- no model/checkpoint selection, active-purity claim, self-exclusion claim, or
  Goal acceptance was recorded;
- unknown/unlabeled objects were not assigned background or Novel labels.

The future probe is only a first-video suffix mutation, not an exhaustive
all-video causal proof. The receipts therefore remain development evidence,
not a qualified M result.

## Measured prefix-16 development values

The following aggregates are over all 27 order probes, without selecting a
variant or seed:

| Quantity | Mean | Min | Max |
|---|---:|---:|---:|
| Known macro accuracy | 0.019556 | 0.010667 | 0.032000 |
| Known micro accuracy | 0.024283 | 0.013245 | 0.039735 |
| Certified cross-video reuse macro lower bound | 0.074562 | 0.066220 | 0.082435 |
| Certified cross-video reuse micro lower bound | 0.094616 | 0.067227 | 0.109244 |
| Physical commit-track coverage | 1.000000 | 1.000000 | 1.000000 |

Mean physical actions over the same 27 probes were `8458.33 EXISTING`,
`68.19 KNOWN`, and `226.48 NEW`. These low Known values and the lack of active
Pred/self-excluded memory are evidence against accepting any of the three M
controls as the research solution.

## Current research consequence

This closes the recovery-2 receipt-integrity question, not CVPR-4 or the
overall goal. `Goal_A`, `Goal_B`, and `Goal_C` remain pending; formal Pred and
final results remain null. The next valid stage is to freeze the complete joint
causal protocol, then add separately trained/self-correcting M controls and a
lawful strong Pred/dense frontend endpoint before any formal development or
final evaluation. No model is selected from this matrix.
