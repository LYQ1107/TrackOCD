# TrackOCD Phase 88 — Persistent OCD Contract Repair + Full Runtime Training

**Status:** resource-blocked before the formal C0v2 gate
**Decision:** `P88_C0V2_FORMAL_INCOMPLETE_RAM_FLOOR_STOP_NO_CONTROLLER`
**Registered window:** 2026-09-07 06:20:15–16:20:15 UTC
**Repository at registration:** `3c12af72a83681883cd6e4e5ec0beeb35d78bdd3`
**Report generated:** 2026-09-07 (Asia/Shanghai)

This report records the complete Phase88 work that actually ran. It does not convert a smoke result, a partial checkpoint, or an old diagnostic replay into a formal MOT+OCD result. The formal C0v2 training route was stopped for the mandatory RAM safety boundary after three bounded repair attempts. Consequently C0v2 formal validation, the C1 support continuation, unchanged-controller compatibility, and sealed evaluation were not run.

## 1. Scope and sealed boundary

Phase88 was isolated under:

- source: `src/iclr27_phase88/`
- scripts: `scripts/iclr27_phase88/`
- configuration: `configs/iclr27_phase88/`
- documentation: `docs/iclr27_phase88/`
- outputs: `outputs/iclr27_phase88 -> /data2/usr_for_deadline/trackocd_phase88/project_outputs`

Phase87 files were read-only inputs. The registered evaluator was the exact `metrics` implementation from `src/iclr27_phase19r/evaluation/internal.py`. The event denominator remained 76 positive and 76 negative events; no threshold, seed, row key, candidate order, parent assignment, or evaluator denominator was changed.

No DEV+, Q1, public new-model, or sealed labels were accessed. Category/video/track identifiers were used only as TRAIN loss metadata and scoring metadata. They were not model-input tensors. No future rows/tracks, text, category names, semantic IDs, or physical IDs were used as inference features.

The formal thresholds registered before training were:

| metric | threshold |
|---|---:|
| persistent Commit-CT | ≥15/76 |
| category coverage | ≥5 |
| video coverage | ≥8 |
| existing precision | ≥0.70 |
| negative false merge | ≤0.15 |
| known micro | ≥0.206 |
| known macro | ≥0.139 |

These thresholds were not relaxed after observing the partial results.

## 2. Why Phase88 was registered

The corrected Phase87 full-runtime replay exposed a contract mismatch: the previous headline numbers were not reproduced when the Phase19R metric implementation, known branch, source-before-target order, and real persistent state memory were used together. Phase88 therefore changed the runtime contract rather than changing the benchmark:

1. use the exact Phase19R metric source;
2. process every source track before the target track;
3. use a real transactional `StateMemoryV2` with multi-prototype states;
4. include explicit NEW/EXISTING/KNOWN/DEFER/RESET actions and reset targets;
5. include a known-category training branch (25% of TRAIN steps);
6. sample 2–4 causal source tracks and visual hard negatives from different category/video groups;
7. use the same runtime transitions for training and evaluation;
8. retain a raw 768-D visual anchor and forbid identity/text shortcuts.

The implementation is a causal track encoder plus a relation encoder and controller. `support_evidence.py` uses eight causal support features (raw best/second similarity, margin, candidate count, source length, variance, quality, and history consistency). The semantic memory is separate from physical bookkeeping; semantic actions never modify physical track IDs.

## 3. TRAIN event manifest and contract audit

`build_causal_events_v2.py` created the atomic manifest `outputs/iclr27_phase88/manifests/causal_event_v2_manifest.json` and per-fold JSONL files.

| fold | fit events | positive | negative | validation events | target tracks |
|---:|---:|---:|---:|---:|---:|
| 0 | 270 | 114 | 156 | 2500 | 39 |
| 1 | 4397 | 2141 | 2256 | 1396 | — |
| 2 | 4342 | 2114 | 2228 | 1354 | — |
| 3 | 4547 | 2223 | 2324 | 1304 | — |

Each episode uses 2–4 source tracks. Source videos are selected by legal causal order rather than numeric video ID ordering. Visual hard negatives are the highest-cosine different-category/different-video tracks available in the TRAIN fold. A fixed 10% TRAIN-only wrong-binding augmentation supplies RESET targets; validation has no reset augmentation. Fold0 has only 270 legal events, below the desired 500-event diagnostic target; that scarcity is recorded, not hidden by duplication.

The contract check passed:

- exact formal metric source: `src/iclr27_phase19r.evaluation.internal.metrics`;
- known categories: 48, active fold0 known slots: 7;
- known prototype hash: `3282127d7e901332a9b0080910365d7a8c924ca622d04fa92c0a7b586cf78704`;
- all source tracks processed before target: true;
- real StateMemoryV2 used: true;
- RESET targets present: 20 in the contract smoke;
- finite loss and non-zero gradients: true;
- forbidden model-input fields: false.

Artifact: `outputs/iclr27_phase88/audit/contract_checks.json`.

## 4. Corrected replay of Phase87 checkpoints

Before training, old C0/C1 checkpoints were replayed through the corrected Phase88 runtime on all 76 positive and 76 negative diagnostic events. This was a diagnostic compatibility check, not model selection. It did not reproduce the earlier Phase87 headline:

| route | fold Commit-CT | aggregate Commit-CT | notable evidence |
|---|---|---:|---|
| old C0 | 0/12, 0/12, 0/24, 1/28 | 1/76 | fold2 negative false merge 1.0; fold3 known micro/macro .248/.260 |
| old C1 | 0/12, 0/12, 0/24, 0/28 | 0/76 | fold2 negative false merge .1667 |

The full records are in `outputs/iclr27_phase88/audit/phase87_full_runtime_replay.json`. Known metrics are numeric in every fold; no `NA` values were substituted. This replay demonstrates why an exact runtime/evaluator contract is necessary, but it is not a result for the new C0v2 model.

## 5. Smoke and targeted training

The first smoke invocation failed with an undefined `support_mode` argument. The failed marker/log were retained. The smallest signature repair passed:

| run | updates | loss first → last | mean last 100 | mean grad norm | reset targets | result |
|---|---:|---:|---:|---:|---:|---|
| `c0v2_smoke_fix1_f0` | 100 | 3.9100 → 1.9280 | 1.9231 | 3.6615 | 1213 | PASS |
| `c0v2_targeted_f0` | 500 | 3.9100 → 1.7835 | 1.3463 | 4.0010 | 6099 | PASS |

Both runs used bf16 autocast on GPU5, produced finite checkpoints, and exercised RESET supervision. The checkpoint hashes are listed in the decision artifact.

## 6. Intermediate validation of a partial checkpoint

The exact full validation path was first attempted on `c0v2_repair1_f0_step006000.pt`. It recomputes the known stream for each event and ran for more than ten minutes without an atomic artifact. Task-owned PIDs 14241/14243 were stopped; the failure log and empty validation directory remain. This is an evaluator performance failure, not a scientific metric.

A bounded 100-event TRAIN-disjoint diagnostic then completed on GPU5. It is explicitly not comparable to the 76-event formal gate:

| metric | fold0 step6000, 100-event diagnostic |
|---|---:|
| positive events in sample | 51 |
| Commit-CT | 2/51 |
| existing precision | .1818 |
| category coverage | 2 |
| video coverage | 2 |
| negative false merge | .1429 |
| premature rate | .73 |
| duplicate births | 24 |
| known micro/macro | 1.0 / 1.0 |

The high premature rate and low reuse signal were not treated as evidence of a passed route. Artifact: `outputs/iclr27_phase88/validation/c0v2_repair1_f0_partial6000_bounded100/step_006000_metrics.json`.

## 7. Formal C0v2 training and resource events

The registered formal target was 20,000 updates per fold. All launches were bounded and used only task-owned PIDs. External InterMOT/MASA processes were not terminated.

1. **Initial four-worker launch:** GPUs5–8, PIDs 25608–25611, supervisor 25602. At about 6k updates, available RAM fell to approximately 21 GiB. All five task-owned processes were explicitly stopped. Partial 2k–6k checkpoints and `.launched` markers were retained.
2. **Repair1 single worker:** fold0/GPU5, worker 9683, supervisor 9669. It resumed from fold0 step4000 and wrote `c0v2_repair1_f0_step006000.pt`; it was stopped after the bounded continuation/window performance check. No `.done` marker was written.
3. **Repair2 pair:** workers 39839/39844 on GPUs5/6, supervisor 39830. Available RAM fell to approximately 27 GiB and no new checkpoint was produced. Only those three task-owned PIDs were stopped; both `.launched` markers remain.
4. **Evaluator performance attempt:** task-owned PIDs 14241/14243 were stopped after the exact full validation path exceeded ten minutes without an artifact. No model output was selected.
5. **Repair3 single worker:** GPU5, worker 30223, supervisor 30216 and wrapper 30214. Fold0 resumed from step6000 and atomically saved `c0v2_repair3_f0_step008000.pt`. Available RAM then fell to approximately 31 GiB, below the 25% safety floor for the 125 GiB host. These three task-owned PIDs were explicitly stopped. After cleanup, available RAM recovered to ~36 GiB and GPU5 was idle.

The third bounded repair/resource attempt exhausts the C0v2 route under this host-memory condition. No broad kill was used. No OOM kernel event occurred; all stops were deliberate safety interventions. The full C0v2 formal 20k/fold route therefore has no `.done` units and no formal validation metrics.

Retained checkpoint:

`outputs/iclr27_phase88/checkpoints/c0v2_repair3_f0_step008000.pt`
SHA256 `81956016f8f73de545780d1852fa565b2002a18684e3552e2f48a8a801d2b30c`

Other retained hashes:

- smoke: `69868ddb13a0aa82bcb02981f3b434d3a9bc6322a468f47237615a3a6db66ada`;
- targeted: `9a2d89ff13954865747ed00cc4ed2a139b4bc1e20673b6832dc57d14209a5247`;
- repair1 fold0 step6000: `efd0d418c86296d10555d7745f838097668213af4cffd9fbb99b9024484556aa`.

## 8. Gate decisions

| gate | status | reason |
|---|---|---|
| contract | PASS | exact metric source, causal source-before-target order, real StateMemoryV2, reset targets, no forbidden inputs |
| event manifest | PASS with fold0 scarcity | legal TRAIN event construction complete; fold0 has 270 fit events |
| Phase87 corrected replay | diagnostic complete | old checkpoints do not reproduce the prior headline |
| C0v2 smoke | PASS | finite bf16 loss and non-zero gradients |
| C0v2 fold0 targeted | PASS | 500 updates, finite checkpoint and RESET sampling |
| C0v2 formal 20k | INCOMPLETE | repeated RAM-floor stops; only partial f0 reached step8000 |
| C0v2 formal gate | NOT EVALUATED | no complete four-fold validation artifact |
| C1 support continuation | NOT STARTED | C0v2 formal checkpoint was not completed |
| unchanged controller compatibility | NOT RUN | no authorized frozen candidate |
| sealed evaluation | SEALED | no DEV+/Q1/public/sealed labels accessed |

The partial 2/51 diagnostic is not a formal success or failure classification. It cannot be compared directly with the registered 15/76 threshold. Since the formal candidate was never completed, claiming a C0v2 gate result would be scientifically invalid.

## 9. Reproduction commands

From the repository root:

```bash
python scripts/iclr27_phase88/register_phase88.py
python scripts/iclr27_phase88/build_causal_events_v2.py
python scripts/iclr27_phase88/contract_checks.py
python -m py_compile scripts/iclr27_phase88/*.py src/iclr27_phase88/*.py
bash scripts/iclr27_phase88/run_c0v2_repair3_supervisor.sh
python scripts/iclr27_phase88/evaluate_checkpoint_v2.py \
  --fold 0 \
  --checkpoint outputs/iclr27_phase88/checkpoints/c0v2_repair3_f0_step008000.pt \
  --tag c0v2_repair3_f0_partial8000_bounded100 \
  --device cuda:5 --max-events 100
```

The last command is a bounded diagnostic only. A future continuation must not relaunch any unit with an existing `.launched` marker; it must use an explicit new authorized route and an isolated memory-safe host or equivalent implementation repair.

## 10. Integrity and limitations

- Phase88 output symlink resolves to `/data2/usr_for_deadline/trackocd_phase88/project_outputs`; large artifacts were not copied into `/data1`.
- Existing Phase87 files and metrics were not overwritten.
- All failed markers, logs, partial checkpoints, and resource events are retained.
- No Phase88 `.done` formal fold marker exists; this is intentional and reflects incomplete training.
- No MOT physical stream was modified by Phase88; no new MOT HOTA/IDF1 result is claimed.
- No controller, StateMemory compatibility replay, persistent Commit-CT result, or sealed evaluation was run for Phase88.
- The Phase87 replay is historical diagnostic evidence only, not a Phase88 outcome.
- The host's external memory pressure is the actionable blocker. The scientific question remains open; the route was not stopped because the task was proven infeasible.

Machine-readable decision and all hashes are in `outputs/iclr27_phase88/audit/final_decision.json`. The next legitimate action is to obtain an explicitly authorized memory-safe execution environment (or implement a measured memory-bounded data/rollout path), then complete the unchanged 20k/fold C0v2 protocol before opening C1. It is not legitimate to run controller tuning, threshold sweeps, a new backbone, or sealed evaluation from the partial checkpoint.
