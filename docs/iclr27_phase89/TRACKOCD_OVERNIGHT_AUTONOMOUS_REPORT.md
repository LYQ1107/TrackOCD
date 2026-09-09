# TrackOCD Phase88R → Phase89 Overnight Autonomous Report

Generated 2026-09-10 (Asia/Shanghai). This report is the corrected completion record for the authorized overnight route; it does **not** claim full MOT+OCD or sealed success.

## 1. Provenance, scope, and sealed boundary

- Repository: `https://github.com/LYQ1107/TrackOCD.git`.
- Starting source head: `58ced135cf75b058774403b32fc6e8ef54a0deed`.
- New work is isolated in `src/iclr27_phase89/`, `scripts/iclr27_phase89/`, `outputs/iclr27_phase89/`; Phase88 historical outputs are preserved.
- TRAIN-only data were used for training and checkpoint selection. The 76-positive/76-negative replay and true held OLD/NEW stream are reporting diagnostics only.
- DEV+, Q1, public new-model labels and sealed labels were not accessed. No future rows/tracks, category text, semantic IDs or physical IDs were model inputs.
- Physical detection/tracking, row keys, event denominator, action masks, thresholds and support protocol were not changed.

## 2. Protocol correction

The old Phase88R H2 headline `4/76` was not a valid H2 attribution. The old replay forced `support_mode=true` although H2_EQUAL registered `support_mode=false`; the old selection also compared H2 30k against a 20k C0 endpoint. The old artifacts remain untouched and are explicitly invalidated for attribution in `outputs/iclr27_phase88/audit/phase88r_protocol_correction_v2.json`.

The corrected route compares C0_CONTINUE 20k→30k to H2_EQUAL 20k→30k with the same seed (`88002`), `fix2` manifest, optimizer/data protocol, and support disabled. Fold-0's missing sampler/rollout state is matched by the existing deterministic continuation policy (`F0_MATCHED_CONTROL_ALREADY_EXISTS`), so no unregistered retraining was introduced.

## 3. Corrected H2 evidence

| fold | C0 continuation | H2 equal | Δ H2−C0 | H2 win |
|---:|---:|---:|---:|:---:|
| 0 | 0.118402 | 0.193641 | +0.075239 | yes |
| 1 | 0.201708 | 0.210676 | +0.008969 | yes |
| 2 | 0.251350 | 0.272499 | +0.021149 | yes |
| 3 | 0.282429 | 0.265279 | −0.017149 | no |
| **mean** | **0.213472** | **0.235524** | **+0.022052** | **3/4** |

Corrected no-support H2 held diagnostic: Commit-CT `10/76`, category/video coverage `7/9`, existing precision/recall/F1 `0.4828/0.1923/0.2750`, new precision/recall `0.4324/0.6316`, open-world false assignment `0.3684`, premature `0.6053`, unresolved `0`, duplicate births `115`, known micro/macro `1.0/1.0`. This is a frozen diagnostic, not a final result.

Corrected TRAIN pseudo-novel stream is reporting-only: all accuracy micro/macro `0.3383/0.4635`, old `0.8519/0.8269`, pseudo-novel `0.3133/0.4538`, H-score `0.4581/0.5330`, NMI macro/weighted `0.4717/0.3467`, ARI macro/weighted `0.2576/0.0755` (589/1741 correct). It uses fold-local Hungarian matching and does not define official held NEW accuracy.

TRAIN evidence still triggered the registered H3 condition: open-world false assignment, duplicate births, premature decisions and insufficient correct reuse remained materially high. No held number was used to trigger H3.

## 4. H3 hierarchical open-world controller

H3 replaces one uncalibrated joint action softmax with a causal hierarchy:

```text
track/history/quality + known summary + memory summary
                 ↓
       [KNOWN | OPEN | DEFER | RESET] router
          ↓ KNOWN             ↓ OPEN
  active-known head       [EXISTING | NEW] head
```

The router consumes only causal track state, quality, known-prototype summary, memory evidence/age/dispersion and zero support features (`support_mode=false`). Physical IDs remain bookkeeping only. H3 reuses the validated Phase88 encoder/memory/transaction semantics and adds router/open-action heads; it emits the legacy-compatible joint action interface. No threshold sweep or StateMemory redesign was performed.

### Equal-budget training

Both H3 and C0_REOPT start from the same `c0v2_fix2_formal_f{0..3}.pt` family at step 20,000, use fresh AdamW over trainable parameters, seed 88002, and run to step 30,000. H3 and C0 are therefore a matched 10k-update comparison. All workers used BF16 when supported, one bounded worker per available/shared GPU, and atomic checkpoints/markers.

| fold | H3 final checkpoint SHA256 | H3 start step | C0 final checkpoint SHA256 |
|---:|---|---:|---|
| 0 | `eabec550a7083c12aec66e96710a6268cd24047d556a84e80ebe2608b858eb88` | 24722 (resumed RAM pause) | `7dac186113ff1d11c4a560fe5d31fbcdf5f9f2cf24d7fe40c93c2c8b4c37a33b` |
| 1 | `384c4ecc49df900a59a400e421b7957889d53dca6c657a826a53bc093fb1e19b` | 23589 (resumed RAM pause) | `2e199bb10393d9441fc88b7f4096eff2f5c3e6a654eca3ec7c91165fe7c65d78` |
| 2 | `e4df1c2c9beb449a3ede6a0a37e80d911e9516aeec6c4f902eafe580c228d209` | 23388 (resumed RAM pause) | `01f21d5ce6f4ced6f5cf11c1bbb9a8404fd143de4c893fd396977b745d32eb30` |
| 3 | `d404bbd31413d7ccf7456d63315934cc72e254abd35c6b9c675383ed721afd66` | 23471 (resumed RAM pause) | `56c3b5dd4d66823ca8c403222c20dd1f143a6271f8776b8bb06f4ce9eceaed49` |

Two RAM-safety pauses occurred. The first supervisor implementation stranded paused units; a second execution audit found that clean worker exit after `SIGUSR1` was not re-queued. Both were repaired minimally, with the exact paused checkpoint/step resumed. No worker was restarted from 20k, no external PID was touched, and all four H3 units ultimately reached `.done` at 30k. `MemAvailable` (not `MemFree`) was the sole pause criterion; the 31.25 GiB floor and three-sample rule were preserved.

## 5. TRAIN-disjoint H3 selection

| fold | C0_REOPT score | H3 score | Δ H3−C0 | H3 Commit-CT recall | H3 open-world false assignment | H3 premature |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.1000 | 0.1645 | +0.0645 | 0.1473 | 0.6142 | 0.6088 |
| 1 | 0.2056 | 0.2304 | +0.0248 | 0.4864 | 0.6516 | 0.7006 |
| 2 | 0.2351 | 0.2450 | +0.0099 | 0.4297 | 0.5986 | 0.4845 |
| 3 | 0.2623 | 0.2201 | −0.0422 | 0.3585 | 0.5314 | 0.4686 |
| **mean** | **0.2008** | **0.2150** | **+0.0142** | — | — | — |

H3 wins 3/4 folds and is selected by the preregistered mean exact TRAIN-disjoint selection score. This is a selection result, not an OCD headline.

## 6. True held OLD/NEW stream (reporting-only)

The frozen H3 checkpoint was evaluated once on the causal true-held OLD/NEW manifest. Aggregate metrics:

| metric | H3 frozen held stream |
|---|---:|
| rows / all correct | 202 / 115 |
| all accuracy micro / macro | 0.5693 / 0.5607 |
| OLD accuracy micro / macro | 0.9753 / 0.9543 |
| held NEW accuracy micro / macro | 0.2975 / 0.3055 |
| H-score micro / macro | 0.4560 / 0.4606 |
| NMI macro / weighted | 0.7667 / 0.7770 |
| ARI macro / weighted | 0.1067 / 0.1027 |

The stream is `FROZEN_HELD_REPORTING_ONLY`, not a checkpoint-selection or threshold-tuning result. H3 improves the corrected H2 diagnostic count to `21/76` on the separate 76+76 replay (category coverage `10`, video coverage `18`), but safety remains inadequate: negative false assignment `36/76=0.4737`, premature `0.6053`, duplicate births `92`, existing precision/recall `0.4974/0.2582`, known capture error `0.1579`, and known micro/macro `0.9692/0.9861`. Per-fold Commit-CT is `[1/12, 8/12, 5/24, 7/28]`; this is not broad safety-preserving success.

## 7. Physical compatibility and final gates

The requested Q0-vs-temporal physical compatibility adapter is not available in the frozen Phase26/Phase88 interface: it would require a new mapping from full raw-video physical stream state to H3 without a registered physical evaluator change. The honest artifact is `outputs/iclr27_phase89/audit/physical_compatibility.json` with status `PHYSICAL_TO_OCD_ADAPTER_NOT_AVAILABLE`; no HOTA/DetA/AssA/MOTA/IDF1 or physical Commit-CT values are invented.

| gate | status | reason |
|---|---|---|
| corrected H2 TRAIN selection | PASS (selection only) | H2 mean > matched C0, 3/4 folds |
| H3 TRAIN selection | PASS (selection only) | H3 mean 0.21499 > C0_REOPT 0.20075, 3/4 folds |
| H3 held diagnostic | DIAGNOSTIC | 21/76, but false assignment/premature/duplicates remain high |
| physical MOT compatibility | NOT RUN | no legal frozen adapter; no standard MOT claim |
| public / Q1 / sealed | SEALED | no access |
| complete MOT+OCD | **NOT COMPLETE** | physical full-sequence compatibility and safety-preserving persistent OCD are unproven |

The route does not authorize another threshold/memory/backbone lottery. The scientific result is that hierarchical arbitration materially improves the matched TRAIN score and yields nonzero cross-fold held CT, but the final causal safety contract is still not met. The remaining bottleneck is joint physical-to-semantic interface and calibrated negative/open-world evidence, not a missing held-label tuning pass.

## 8. Reproduction and artifacts

```bash
/home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase89/run_h3_multigpu.py --route h3_router
/home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase89/run_h3_multigpu.py --route c0_reopt
/home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase89/run_validation.py --route h3_router
/home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase89/run_validation.py --route c0_reopt
/home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase89/select_h3.py
TRACKOCD_OUT=outputs/iclr27_phase89 /home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase88/evaluate_frozen_76_diagnostic.py --selection outputs/iclr27_phase89/audit/h3_frozen_selection.json --tag h3_router_76plus76_diagnostic --support-mode auto --architecture h3 --device cuda:0 --memmap-root /data2/usr_for_deadline/trackocd_phase88/shared_features
TRACKOCD_OUT=outputs/iclr27_phase89 /home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase88/evaluate_standard_gcd_stream.py --selection outputs/iclr27_phase89/audit/h3_frozen_selection.json --support-mode auto --architecture h3 --device cuda:0 --manifest outputs/iclr27_phase88/manifests/held_standard_openworld_stream_v1.json --output outputs/iclr27_phase89/audit/held_standard_openworld_metrics_h3.json --status FROZEN_HELD_REPORTING_ONLY
```

Key machine-readable artifacts:

- `outputs/iclr27_phase88/audit/phase88r_protocol_correction_v2.json`
- `outputs/iclr27_phase88/audit/h2_equal_train_selection_corrected.json`
- `outputs/iclr27_phase88/diagnostic/h2_equal_76plus76_corrected_nosupport/metrics.json`
- `outputs/iclr27_phase88/audit/standard_gcd_stream_metrics_corrected.json`
- `outputs/iclr27_phase89/audit/h3_preregistration.json`
- `outputs/iclr27_phase89/audit/h3_train_selection.json`
- `outputs/iclr27_phase89/audit/h3_frozen_selection.json`
- `outputs/iclr27_phase89/audit/held_standard_openworld_metrics_h3.json`
- `outputs/iclr27_phase89/diagnostic/h3_router_76plus76_diagnostic/metrics.json`
- `outputs/iclr27_phase89/audit/physical_compatibility.json`

No public/sealed result is claimed. The next scientifically justified work would require a separately registered physical-to-OCD adapter/full-sequence evaluator and explicit negative-evidence calibration; it is not legal to infer that from this diagnostic alone.
