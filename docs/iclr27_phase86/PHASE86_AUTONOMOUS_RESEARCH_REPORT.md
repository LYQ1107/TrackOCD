# TrackOCD Phase86 — Autonomous Research Report

## Decision

**Status:** `PHASE86_DIAGNOSTIC_COMPLETE_U1_EVENT_SAFETY_FAIL_RF_NEGATIVE_U2_NOT_OPENED`  
**Diagnostic label:** `DIAGNOSTIC_ONLY_DO_NOT_SELECT`  
**Formal OCD / sealed:** not run.

Window: `2026-09-06T07:57:11.490953Z` → `2026-09-06T17:57:11.490953Z` (registered ten-hour window). Start HEAD: `43a8c4e248f58b81deb0b831084282603e7bf1d6`; Phase85 artifacts were read-only. Large outputs: `/data2/usr_for_deadline/trackocd_phase86/project_outputs` via the `outputs/iclr27_phase86` symlink.

## Frozen boundary and resources

- No DEV+, Q1, public-new, or sealed labels were used as model inputs or for selection. No future rows/tracks, category/text, semantic IDs, or physical IDs entered inference tensors.
- GPU0 external PID 33785 was not touched. Phase86 diagnostic/U1/RF work was CPU-only; no task GPU worker, OOM, or external-process termination occurred. RAM was about 125 GiB total with about 113 GiB available at registration; `/data1` had about 29 GiB free and `/data2` about 1.1 TiB.
- Phase85 and earlier files remain read-only. Failed markers and hashes are retained; artifacts are atomic where applicable.

## Historical frozen controller

The chronology/formal-use rule selected Phase19R RC-MS-OCD plus its shared StateMemory, with checkpoint/code hashes in `/data2/usr_for_deadline/trackocd_phase86/project_outputs/manifests/frozen_controller_manifest.json`. The selection was made before the Phase86 replay and was not performance-selected. Phase56's controller was not substituted.

## D0–D3 frozen diagnostic OCD

| stream | Commit-CT | fold distribution | status |
|---|---:|---|---|
| D0 historical Q0 + frozen RC-MS-OCD | 3/76 | f0=0/12, f1=0/12, f2=0/24, f3=3/28 | diagnostic only |
| D1 temporal physical + frozen RC-MS-OCD | 3/76 | f0=0/12, f1=0/12, f2=0/24, f3=3/28 | diagnostic only |

D2 raw source-conditioned support and D3 bounded reranker support were not run through the controller: their frozen Phase85 artifacts are score-level evidence, not legal causal 768-D row-vector inputs. No score calibration or invented adapter was used. The Phase86 summary and `event_traces.jsonl` are diagnostic-only. The legacy Phase72 parity comparison is explicitly `protocol_equal=false` because its JSON uses a different event-accounting representation.

## U1 — OOF selective intervention

The frozen Phase85 raw and bounded reranker experts were not retrained. U1 used OOF TRAIN predictions only, with a fixed 64-D causal score/context feature vector, MLP `64→32→utility`, labels HELP=+1, HARM=-4, both-correct/both-wrong=0, and `utility>0` selecting reranker else exact raw. The 2,095-row meta manifest and expert/checkpoint hashes are recorded under `outputs/iclr27_phase86/manifests/`.

| OOF validation fold | rescue | harm | net rescue | TRAIN gate |
|---:|---:|---:|---:|---|
| 0 | 2 | 1 | 1 | PASS |
| 1 | 3 | 1 | 2 | PASS |
| 2 | 2 | 3 | -1 | FAIL |

TRAIN gate: **PASS** (at least 2/3 folds satisfied net rescue>0 and harm≤rescue).

| p16 event replay | raw | full reranker | U1 selective | registered status |
|---|---:|---:|---:|---|
| positive reliable events | 20/76 | 23/76 | 22/76 | ≥22 |
| negative reliable events | 8/76 | 15/76 | 12/76 | ≤8 |

U1 event status is **FAIL**: selective intervention reaches 22/76 positives but 12/76 negatives, and uses the reranker on 58/76 negative events. Consequently no controller compatibility or Commit-CT claim was authorized. This event replay is a score-level route and does not override the U1 TRAIN trigger rule.

## RF — Root-of-Fragments diagnostic

RF was parameter-free: fixed causal contiguous four-frame fragments from frozen Phase75D vectors, symmetric Chamfer matching, and raw + `0.05*tanh` residual. It produced p16 R@1 `0.893219` vs raw `0.893219`, mAP `0.848641` vs raw `0.848374`, hard-gap `0.188156` vs raw `0.189559`, unsafe flips `0`, and `2/4` non-decreasing folds. Decision: **ROOT_OF_FRAGMENTS_V1_NEGATIVE**. No training or controller run followed.

## Failures and repairs

- Diagnostic r0: import path failure; retained in `completion/diagnostic_ocd_r0_failed.json`; fixed by the minimal project-root path insertion.
- Diagnostic r1: Phase85 artifact path resolved under Phase86 output; D0/D1 completed in memory but summary commit failed; retained in `diagnostic_ocd_r1_failed.json`; fixed by a path-only correction.
- U1 r1 smoke: scalar/list counter initialization raised a TypeError before metrics; retained as a failed marker; fixed by typed counters and rerun with fresh tag r2. U1 smoke, targeted, and formal artifacts are separate and atomic.
- No OOM, duplicate supervisor, broad kill, or external-process termination occurred.

## Gate and next-route decision

| gate | result | evidence |
|---|---|---|
| D0/D1 diagnostic | COMPLETE | 3/76 each, all fold3; diagnostic only |
| U1 TRAIN safety | PASS | 2/3 OOF folds pass |
| U1 event safety | FAIL | 22/76 positive, 12/76 negative |
| RF TRAIN validation | FAIL | 2/4 non-decreasing; no safe improvement |
| controller/formal OCD | NOT RUN | no legal safe upstream candidate |
| sealed/public | NOT RUN | sealed boundary preserved |

U2 was **not opened**: the pre-registered trigger is a failed U1 TRAIN gate, while U1 TRAIN safety passed on 2/3 folds. Opening U2 solely because the event score route failed would silently change the registered route order. A future U2 requires explicit authorization or a new preregistration.

## Reproduction commands

```bash
cd /data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
/home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase86/discover_controller.py
/home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase86/run_diagnostic_ocd.py
/home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase86/validate_baseline_parity.py
/home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase86/run_u1_selective.py --mode smoke --tag r2 --fold 0
/home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase86/run_u1_selective.py --mode targeted --tag r1 --fold 0
for f in 0 1 2; do /home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase86/run_u1_selective.py --mode formal --tag r1 --fold "$f"; done
/home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase86/run_u1_event_replay.py --tag u1_formal_replay_r1
/home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase86/run_rf_diagnostic.py
```

## Key artifacts

- Report: `/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/docs/iclr27_phase86/PHASE86_AUTONOMOUS_RESEARCH_REPORT.md`
- Decision: `/data2/usr_for_deadline/trackocd_phase86/project_outputs/audit/phase86_decision.json`
- Diagnostic: `/data2/usr_for_deadline/trackocd_phase86/project_outputs/diagnostic_ocd/summary.json`
- U1 decision: `/data2/usr_for_deadline/trackocd_phase86/project_outputs/audit/u1_decision.json`
- RF decision: `/data2/usr_for_deadline/trackocd_phase86/project_outputs/audit/rf_decision.json`
- Controller manifest: `/data2/usr_for_deadline/trackocd_phase86/project_outputs/manifests/frozen_controller_manifest.json`

The report does not claim complete MOT+OCD. Persistent Commit-CT formal and sealed evaluation remain unrun because the only registered score-level intervention failed event safety and no controller-compatible safe upstream candidate was produced.
