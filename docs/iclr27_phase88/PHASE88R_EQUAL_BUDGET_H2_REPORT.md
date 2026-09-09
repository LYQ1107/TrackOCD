# TrackOCD Phase88R — Equal-Budget H2 Report

Generated: 2026-09-09T16:18:52.247295+00:00

## Decision

**H2_EQUAL_BUDGET TRAIN selection: SELECTED. Held diagnostic: completed but not a broad final success.**
The registered H2 route won the exact TRAIN-disjoint selection score, so one frozen 76-positive + 76-negative diagnostic was allowed. It produced 4/76 Commit-CT, but all four correct events are from fold0 (category coverage 2, video coverage 3); no public/Q1/sealed evaluation was run.

Machine-readable artifacts:
- `outputs/iclr27_phase88/audit/hypothesis_2_equal_budget_preregistration.json`
- `outputs/iclr27_phase88/audit/resource_metric_correction.json`
- `outputs/iclr27_phase88/audit/h2_equal_train_selection.json`
- `outputs/iclr27_phase88/audit/h2_equal_frozen_selection.json`
- `outputs/iclr27_phase88/diagnostic/h2_equal_76plus76_diagnostic/metrics.json`
- `outputs/iclr27_phase88/audit/standard_gcd_stream_metrics.json`

## Frozen protocol and scope

- Four fixed TRAIN video/category-disjoint folds; event_tag `fix2`; seed 88002; support_mode false.
- Each H2 fold resumed its same-fold C0v2 fix2 checkpoint at step 20,000 and ended at step 30,000 (+10,000 updates). Only the known-suppression loss term (weight 1.5) changed.
- Optimizer, model tensors, causal event manifest, row order, evaluator, denominator and action semantics were retained. No threshold, StateMemory, physical tracker, detector or backbone change was made.
- TRAIN validation was used for selection. The held replay was run once after freezing and was not used to tune or select anything.
- Public DEV+, Q1, public new-model labels and sealed labels were not accessed; future rows/tracks, category text, semantic IDs and physical IDs were not model inputs.

## Historical frozen comparators (not H2 selection inputs)

The earlier same-protocol TRAIN-only comparators are retained for context: C0 continuation mean selection **0.213472**, C1 support mean **0.200465**, and H1 false-merge/reset mean **0.210993**. The earlier C0 held diagnostic was **2/76** with existing precision **1.0**; those held outcomes were not used to select H2. The old extended-budget H2 artifacts remain historical/diagnostic evidence and were not used to initialize the equal-budget route.

## Resource-protocol repair

The old Phase88 H2 stop artifacts used `MemFree` as the hard trigger. Under shared NumPy memmaps this is not a valid safety estimate because reclaimable page cache can reduce MemFree while MemAvailable remains high. Phase88R records both fields but gates only on `MemAvailable < 31.25 GiB` for three consecutive 30-second samples.

The equal-budget supervisor discovered four usable GPUs and ran one bounded worker per fold (GPU0–3 at launch). No external PID was touched. MemFree reached below 1 GiB during the run while MemAvailable stayed above 70 GiB; no OOM or resource pause occurred. A two-step resume smoke passed the expected start-step check and produced atomic checkpoint/metrics.

The f0 C0 base checkpoint predates sampler/rollout-state persistence. This is explicitly marked `resume_state_repair_required=true` in the f0 H2 metrics and preregistration; it is not presented as an exact RNG continuation. f1–f3 restored sampler and RNG state. Historical extended-budget H2 checkpoints remain diagnostic-only and did not initialize this route.

## Equal-budget training

| fold | start→final | loss first→last | checkpoint SHA256 | resume-state repair |
|---:|---:|---:|---|---|
| 0 | 20000→30000 | 0.137491→0.006241 | `9eb5b6928edbb568e4e6c9dbcce83d8bfe9072853f1962836d0cf133a8323d6f` | `True` |
| 1 | 20000→30000 | 0.764431→3.793005 | `fb6fbf87fa683e1f08f8fe33be78d28254afa1df8d232322941ed4b7e245bb8c` | `False` |
| 2 | 20000→30000 | 0.134115→0.130791 | `da571b9d78b59064b54a3ace606d8d5b83d4f1f3c3e3dedadee83d53fb81b8ee` | `False` |
| 3 | 20000→30000 | 0.457885→0.086158 | `1921be8e51401f802758571a93215af66265114dfb0337646746dc7c3f18a153` | `False` |

## TRAIN-disjoint validation and selection

| fold | C0 selection | H2 selection | Δ H2−C0 | H2 CT recall | H2 neg false merge | category/video coverage |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.103983 | 0.193641 | 0.089657 | 0.1612 | 0.7183 | 13/109 |
| 1 | 0.203037 | 0.210676 | 0.007639 | 0.4849 | 0.6817 | 5/71 |
| 2 | 0.249560 | 0.272499 | 0.022938 | 0.4755 | 0.5957 | 8/46 |
| 3 | 0.250234 | 0.265279 | 0.015046 | 0.4434 | 0.5629 | 9/38 |

C0 mean selection score: **0.201703506**; H2 mean: **0.235523703**; H2 wins 4/4 folds. This is the preregistered selection result.

## One held 76+76 diagnostic

| metric | aggregate |
|---|---:|
| commit_ct | 4/76 |
| existing_precision | 0.592593 |
| existing_recall | 0.043956 |
| existing_f1 | 0.081841 |
| negative_false_merge_rate | 0.157895 |
| anonymous_false_merge_rate | 0.039474 |
| known_capture_error_rate | 0.118421 |
| open_world_false_assignment_rate | 0.157895 |
| premature_rate | 0.605263 |
| unresolved_rate | 0.000000 |
| duplicate_births | 128 |
| category_coverage | 2 |
| video_coverage | 3 |
| known_micro | 1.000000 |
| known_macro | 1.000000 |

| fold | Commit-CT | category coverage | video coverage | negative false assignment |
|---:|---:|---:|---:|---:|
| 0 | 4/12 | 2 | 3 | 0.583333 |
| 1 | 0/12 | 0 | 0 | 0.166667 |
| 2 | 0/24 | 0 | 0 | 0.000000 |
| 3 | 0/28 | 0 | 0 | 0.107143 |

Interpretation: H2 improves the historical 3/76 count to 4/76 and sharply reduces aggregate false assignment relative to the old C0 diagnostic, but it is concentrated in fold0 and does not establish broad cross-fold OCD. Duplicate births and premature commits remain material. This result is diagnostic-only and does not authorize public/sealed evaluation.

## Auxiliary standard stream metrics

The deterministic `standard_gcd_stream_v1` manifest keeps one StateMemory per fold stream, places known anchors before novel tracks, and never uses category/performance ordering. Anonymous SID tokens are fold-stream-local; global Hungarian is applied only after stream collection. These metrics are reporting-only and were not used for H2 selection.

| metric | aggregate |
|---|---:|
| all_acc | 0.149339 |
| old_acc | 0.370370 |
| new_acc | 0.138554 |
| h_score | 0.201666 |
| nmi | 0.305252 |
| ari | 0.020401 |
| rows | 1741 |
| old_rows | 81 |
| new_rows | 1660 |

## Route status and next authorization

- `H2_EQUAL_BUDGET`: TRAIN selection and one held diagnostic complete.
- H3 hierarchical router was not opened because the registered condition for H3 was H2 TRAIN failure.
- No new controller/backbone/threshold experiment, public evaluation or sealed evaluation was run in Phase88R.
- The full TrackOCD research objective is not claimed complete: broad persistent OCD, physical MOT invariants and sealed validation remain outside this route.
- Existing Phase88 historical artifacts, old extended-budget H2 checkpoints, failed supervisor evidence and storage symlink lineage are retained.

## Reproduction

```bash
python scripts/iclr27_phase88/register_h2_equal_budget.py
python scripts/iclr27_phase88/build_standard_stream_manifest.py
python scripts/iclr27_phase88/run_h2_equal_multigpu.py
python scripts/iclr27_phase88/run_h2_equal_validation.py
python scripts/iclr27_phase88/freeze_h2_equal_selection.py
python scripts/iclr27_phase88/evaluate_frozen_76_diagnostic.py --selection outputs/iclr27_phase88/audit/h2_equal_frozen_selection.json --tag h2_equal_76plus76_diagnostic --device cuda:0 --memmap-root /data2/usr_for_deadline/trackocd_phase88/shared_features
python scripts/iclr27_phase88/evaluate_standard_gcd_stream.py cuda:0
```

All commands preserve the TRAIN/held boundary and write atomic outputs. Report generated from the machine-readable artifacts above.
