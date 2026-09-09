# TrackOCD Phase88R — Protocol-Corrected H2 Report

Generated 2026-09-10 after the registered protocol correction and equal-budget rerun. The earlier `PHASE88R_EQUAL_BUDGET_H2_REPORT.md` is retained as historical evidence but its headline H2 held number is invalid for H2 attribution: its replay forced `support_mode=true`, while the registered H2_EQUAL route is `support_mode=false`, and its selection compared H2 30k against a 20k C0 endpoint.

## Corrected decision

The valid comparison is C0_CONTINUE 20k→30k versus H2_EQUAL 20k→30k, same fold manifests, seed 88002, event tag `fix2`, optimizer/data protocol and `support_mode=false`. Corrected TRAIN selection scores are:

| fold | C0 continuation | H2 equal | Δ H2−C0 | H2 win |
|---:|---:|---:|---:|:---:|
| 0 | 0.118402 | 0.193641 | +0.075239 | yes |
| 1 | 0.201708 | 0.210676 | +0.008969 | yes |
| 2 | 0.251350 | 0.272499 | +0.021149 | yes |
| 3 | 0.282429 | 0.265279 | −0.017149 | no |
| **mean** | **0.213472** | **0.235524** | **+0.022052** | **3/4** |

This is a TRAIN-disjoint selection result only. It does not use held labels and does not by itself establish OCD success.

## Corrected held diagnostic

The same frozen H2 checkpoints were replayed once with propagated `support_mode=false`:

| metric | H2 corrected no-support |
|---|---:|
| Commit-CT | 10/76 |
| Existing precision / recall / F1 | 0.4828 / 0.1923 / 0.2750 |
| New precision / recall / F1 | 0.4324 / 0.6316 / 0.5165 |
| anonymous false merge | 0.2895 |
| known capture error | 0.0789 |
| open-world false assignment | 0.3684 |
| premature | 0.6053 |
| unresolved | 0.0000 |
| duplicate births | 115 |
| category / video coverage | 7 / 9 |
| known micro / macro | 1.0000 / 1.0000 |

Per-fold Commit-CT is `[2/12, 1/12, 2/24, 5/28]`; the result is diagnostic-only and is not used to choose H3 or any threshold.

## Corrected auxiliary stream

The TRAIN pseudo-novel stream is explicitly labeled `PSEUDO_NOVEL_TRAIN_REPORTING_ONLY`. It uses fold-local Hungarian matching and count-based micro aggregation (not a cross-fold semantic namespace): all accuracy micro/macro `0.3383/0.4635`, old `0.8519/0.8269`, pseudo-novel `0.3133/0.4538`, H-score `0.4581/0.5330`, NMI macro/weighted `0.4717/0.3467`, ARI macro/weighted `0.2576/0.0755`, 589/1741 correct.

The true held OLD/NEW manifest has no chronology conflicts and is evaluator-side only. It was reserved for the selected H3 route; it was not used in H2 selection.

## Correction provenance and boundaries

Machine-readable correction ledger: `outputs/iclr27_phase88/audit/phase88r_protocol_correction_v2.json`. Old invalid artifacts are preserved. The fold-0 missing sampler/rollout state is matched by the existing C0 continuation deterministic policy (`F0_MATCHED_CONTROL_ALREADY_EXISTS`), so no extra C0 training was needed. No DEV+, Q1, public new-model or sealed labels were accessed; no future rows/tracks, category text, semantic IDs or physical IDs were model inputs. No detector, physical tracker, StateMemory threshold or action denominator was changed.

## Reproduction

```bash
/home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase88/freeze_h2_equal_selection.py \
  --c0-route c0_continue_fix1 \
  --output outputs/iclr27_phase88/audit/h2_equal_train_selection_corrected.json \
  --frozen-output outputs/iclr27_phase88/audit/h2_equal_frozen_selection_corrected.json
/home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase88/evaluate_frozen_76_diagnostic.py \
  --selection outputs/iclr27_phase88/audit/h2_equal_frozen_selection_corrected.json \
  --tag h2_equal_76plus76_corrected_nosupport --support-mode auto
/home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase88/evaluate_standard_gcd_stream.py \
  --selection outputs/iclr27_phase88/audit/h2_equal_frozen_selection_corrected.json \
  --support-mode auto --status PSEUDO_NOVEL_TRAIN_REPORTING_ONLY
```

H2 is not the final TrackOCD result. Its TRAIN absolute open-world errors triggered the preregistered H3 hierarchical route.
