# CVPR-2/3/4 mechanism evidence reconciliation — 2026-10-22

This is an evidence reconciliation, not scientific acceptance. It keeps the
three mechanism directions separate and preserves negative or mixed results.

## R — representation

The fixed matrix contains four variants × three seeds with 200 Train-fit
steps and all twelve Train-development evaluations. At prefix 16, R improves
macro retrieval R1 over the frozen Raw reference in 12/12 comparisons and
macro R5 in 10/12. At `all_available`, the counts are only 2/12 and 10/12.
This is a measurable, prefix-dependent Train-only effect; no checkpoint was
selected and it is not a formal Pred/OCD/MOT result.

## C — candidate-aware decision

The corrected paired matrix has six complete endpoint trials (three seeds ×
three causal orders). At prefix 16, the repair-minus-baseline three-seed mean
changes Known micro accuracy by `-0.0618101545`; certified cross-video reuse
remains `0`. The arm is therefore a complete negative development result for
persistent discovery under this protocol, not a calibrated winner.

## M — persistent memory

The current-source recovery-2 audit verifies nine cases and all three order
probes per case, with no boundary violations. Its 27 prefix-16 rows have mean
Known macro/micro accuracy `0.0195556/0.0242826` and mean certified reuse
lower bounds `0.0745616/0.0946156`. Boundary and future-suffix invariance pass,
but the values are too weak to accept M; active purity, self-exclusion, and
Pred-owned joint evaluation remain open.

## Joint interface and gate

The joint R/C/M pilot proves tensor/lifecycle integration on two fixed policy
videos only; its M agreement loss was legally masked for singleton supports.
It is not a scientific winner. Formal v2 admission remains closed until the
exact frontend provenance, formal predicted stream/cache, and four frozen Val
causal-order assets are recovered. No TAO Test or unauthorized semantic data
was read, and Goals A/B/C remain pending.

Machine receipt: `outputs/trackocd_cvpr/cvpr2/mechanism_evidence_reconciliation_20261022.json`.
