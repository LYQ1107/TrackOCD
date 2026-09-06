# Phase86 Frozen Diagnostic OCD Report

> **DIAGNOSTIC_ONLY_DO_NOT_SELECT** — This replay is an audit of frozen interfaces. Its outputs are not consumed by training, checkpoint selection, threshold selection, or sealed evaluation.

## Scope and boundary

- Controller: chronologically last byte-identifiable formally frozen RC-MS-OCD / Phase19R controller and StateMemory; thresholds, action semantics, and physical stream were unchanged.
- D0: historical Q0 native Phase19R 768-D prefix stream.
- D1: Phase85 temporal physical 768-D prefix vectors with a deterministic causal prefix mapping (1/2/4/8/16).
- D2/D3: retained as explicit interface-incompatible records because their frozen artifacts expose score-level selected-candidate evidence rather than the controller-required causal 768-D row vector. No invented rescaling or adapter was used.
- Denominator: 76 positive events and 76 negative events; no DEV+, Q1, public-new, or sealed labels were read for model input or selection.

## Persistent Commit-CT diagnostic

| stream | correct / eligible | recall | fold results |
|---|---:|---:|---|
| D0_historical_Q0_RCMSOCD | 3/76 | 0.039474 | f0=0/12, f1=0/12, f2=0/24, f3=3/28 |
| D1_temporal_physical_RCMSOCD | 3/76 | 0.039474 | f0=0/12, f1=0/12, f2=0/24, f3=3/28 |

D0 and D1 both produced 3/76, with all three events in fold 3. This is diagnostic evidence only and does not satisfy a broad causal gate. The fold-level parity artifact is intentionally marked non-equivalent to the legacy Phase72 JSON because the historical file used a different per-fold event accounting representation.

## Frozen controller comparison

- Phase86 D0 summary: `/data2/usr_for_deadline/trackocd_phase86/project_outputs/diagnostic_ocd/summary.json` (SHA256 `13488263a5845f7b443760b37974cdbcd4d7b315ee0e785629779e6885ac475b`).
- Historical Phase72 record: `/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/outputs/iclr27_phase72/metrics/phase19r_raw_baseline.json` (SHA256 `aadb4225e3ea794dda7aea371abfc13807bbe493c92b2a47718c56e2b9129465`).
- Parity result: `protocol_equal=False`; this comparison was not used for model selection.

## D2/D3 decision

- D2 raw source-conditioned support and D3 bounded reranker support remain **not interface compatible** with the frozen RC-MS-OCD input contract. Their hashes and reasons are recorded in `summary.json`; they were not silently converted into controller inputs.

## Reproducibility and safety

- The first two failed attempts are retained as `completion/diagnostic_ocd_r0_failed.json` (import path) and `completion/diagnostic_ocd_r1_failed.json` (artifact path). A path-only repair was applied before the successful replay.
- Event traces and the summary were atomically written. No training process was launched by this diagnostic.
- `public_dev_q1_sealed_accessed=false`, `future_rows_or_tracks=false`, and `ids_or_text_as_model_input=false` are asserted in the machine artifact.

## Upstream branch decision

The diagnostic is not an upstream selection result. Per preregistration, Phase86 now audits the TRAIN-only selective intervention (U1) using out-of-fold labels. If its net rescue/harm safety gate fails, the registered order permits one relation-encoder route (U2) and then the frozen fragment representation diagnostic (RF); no threshold, StateMemory, or controller tuning is authorized.

Generated UTC: `2026-09-06T08:10:08.060943+00:00`

