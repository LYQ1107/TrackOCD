# TrackOCD Phase84 — Autonomous Research Report

Status: **AUTONOMOUS_PHASE84_COMPLETE_WITH_INTERFACE_NEGATIVE_EVIDENCE**  
Window: `2026-09-04T17:50:55Z` → `2026-09-05T03:50:55Z`  
Finalized: `2026-09-05T03:05:55.816955+00:00`  
Git HEAD: `9b21041cb63f2f001de15363dab2e621bc0b19a9` (changes were committed and pushed to `origin/main` before report generation)

## Executive decision

Phase84 corrected the Phase83 reporting, physical-lineage, and source/query interface errors. The true full-native physical reassociation completed, but its frozen-R safety gate failed. The repaired query-conditioned native selector and one fixed raw-anchor residual diagnostic did not reach the registered selection criterion. Therefore Phase84 does **not** run alignment, the historical controller, Commit-CT, or sealed/public evaluation. This is a window-level interface/selection negative result, not a claim that TrackOCD is universally infeasible.

## Protocol and sealed boundaries

- Positive/negative event denominator: `76 + 76`; prefixes `(1, 2, 4, 8, 16)`; frozen R universe: `984` queries.
- Physical IDs are runtime bookkeeping only. Inference tensors contain visual/geometry/causal-history fields; no category/text, semantic ID, numeric physical-ID feature, future row/track, held GT, DEV+, Q1, public-new, or sealed label was used.
- Event labels/GT are post-hoc scoring metadata only. No threshold sweep, controller, StateMemory, or sealed/public run occurred.
- Frozen historical Phase75B O reference (reported without reinterpretation): source `49/76`, target `40/76`, both `25/76`.

## Phase83 correction and physical P→R

The A2 report-source audit is `NOT_RUN`; the actual artifact and the mistakenly rendered artifact are retained in `/data2/usr_for_deadline/trackocd_phase84/project_outputs/audit/phase83_a2_report_integrity.json`. No Phase83 artifact was modified.

The true A84P route unions full-native Q0 fragments causally (dormant-only candidates, observed-step timing, gap ≤16, collision-safe canonical IDs), then rebuilds membership before applying the frozen visual R adapter. It preserved the native row denominator and passed Q0 adapter parity. Physical lineage: `682335` rows, `27710` causal unions, `811` multi-category roots among `880` labeled roots; this is post-hoc TRAIN audit evidence, not model input.

### A84P frozen-R prefix comparison

| prefix | queries | physical R@1 | raw R@1 | physical mAP | raw mAP | physical gap | raw gap | unsafe |
|---|---|---|---|---|---|---|---|---|
| 1 | 984 | 0.895334 | 0.886107 | 0.846944 | 0.841994 | 0.168899 | 0.165196 | 3 |
| 2 | 984 | 0.900635 | 0.920327 | 0.860188 | 0.859808 | 0.185398 | 0.181644 | 8 |
| 4 | 984 | 0.905794 | 0.911356 | 0.876884 | 0.856492 | 0.207743 | 0.194318 | 8 |
| 8 | 984 | 0.891683 | 0.906434 | 0.873358 | 0.850146 | 0.204333 | 0.190795 | 11 |
| 16 | 984 | 0.894135 | 0.893219 | 0.871168 | 0.848374 | 0.201346 | 0.189559 | 6 |

At prefix16, physical R@1/mAP are `0.894135`/`0.871168` versus raw `0.893219`/`0.848374`, with `6` unsafe flips. The route gate is **FAIL** (`safe_r_signal=false`); no controller was run.

## B84S source-conditioned support

The corrected same-space TRAIN signal audit was diagnostic only:

| prefix | queries | source-conditioned R@1 | raw R@1 | source-conditioned mAP | raw mAP | source gap | raw gap | unsafe |
|---|---|---|---|---|---|---|---|---|
| 1 | 882 | 0.966557 | 0.908859 | 0.895565 | 0.868716 | 0.188014 | 0.169664 | 3 |
| 2 | 882 | 0.958744 | 0.952665 | 0.907055 | 0.888866 | 0.224683 | 0.192853 | 5 |
| 4 | 882 | 0.974369 | 0.942793 | 0.910003 | 0.888281 | 0.235657 | 0.209748 | 6 |
| 8 | 882 | 0.974369 | 0.944180 | 0.911401 | 0.881842 | 0.250195 | 0.202793 | 4 |
| 16 | 882 | 0.974369 | 0.936698 | 0.910847 | 0.879438 | 0.251195 | 0.202140 | 3 |

The original B84S formal selector used a query-agnostic source attachment and is retained as a failed interface comparator. Its p16 replay summary is `{"negative": {"events": 76, "frozen_both_reliable": 24, "frozen_source_reliable": 49, "frozen_target_reliable": 40, "polarity": "negative", "prefix": 16, "raw_source_mean_reliable_events": 8, "selected_candidate_events": 12, "selected_reliable_events": 0}, "positive": {"events": 76, "frozen_both_reliable": 25, "frozen_source_reliable": 49, "frozen_target_reliable": 40, "polarity": "positive", "prefix": 16, "raw_source_mean_reliable_events": 20, "selected_candidate_events": 33, "selected_reliable_events": 9}}`.

### B84S-Q repaired query contract

The repaired manifest uses legal Phase30 TRAIN query/support pairs, native Q0 candidate sets, explicit DEFER, event-video exclusion, and a deterministic three-fold fallback because a four-fold split could not retain the preregistered minimum fit/validation group counts. It contains `753` groups and `23039` candidate rows; the fold imbalance is retained rather than hidden.

TRAIN-disjoint validation (all completed folds):

| fold | steps | groups | candidate top1 | candidate top5 | candidate/DEFER acc | DEFER recall |
|---|---|---|---|---|---|---|
| 0 | 3780 | 386 | 0.000000 | 0.607143 | 0.515544 | 0.555866 |
| 1 | 16980 | 106 | 0.093023 | 0.930233 | 0.603774 | 0.952381 |
| 2 | 8460 | 261 | 0.093750 | 0.531250 | 0.697318 | 0.781659 |

Frozen event replay (all `760` records):

| prefix | polarity | events | selected candidate | selected reliable | raw source-mean reliable | frozen source | frozen target | frozen both |
|---|---|---|---|---|---|---|---|---|
| 1 | positive | 76 | 76 | 12 | 10 | 49 | 29 | 17 |
| 1 | negative | 76 | 76 | 3 | 2 | 49 | 29 | 18 |
| 2 | positive | 76 | 76 | 16 | 14 | 49 | 35 | 22 |
| 2 | negative | 76 | 76 | 5 | 4 | 49 | 35 | 20 |
| 4 | positive | 76 | 76 | 17 | 15 | 49 | 36 | 22 |
| 4 | negative | 76 | 76 | 7 | 5 | 49 | 36 | 21 |
| 8 | positive | 76 | 76 | 21 | 17 | 49 | 38 | 23 |
| 8 | negative | 76 | 76 | 9 | 8 | 49 | 38 | 22 |
| 16 | positive | 76 | 76 | 24 | 20 | 49 | 40 | 25 |
| 16 | negative | 76 | 76 | 9 | 8 | 49 | 40 | 24 |

At prefix16 B84S-Q selected reliable candidates on `7/76` positive events and `2/76` negative events, versus raw source-mean `20/76` and `8/76`. All event candidate sets were nonempty (median `196.5` candidates). The repaired selector therefore **FAILS** to improve the frozen support selection; this is not an empty-pool result.

### B84S-RA raw-anchor diagnostic

The one registered no-training diagnostic added a fixed `0.05*tanh` bounded residual to raw source-mean cosine and used raw candidate fallback when the frozen model emitted DEFER. At prefix16 it reached `24/76` positive and `9/76` negative reliable selections, versus raw `20/76` and `8/76`. Per-fold positive counts and event taxonomy are in `/data2/usr_for_deadline/trackocd_phase84/project_outputs/audit/b84sra_failure_audit.json`. The modest positive increase remains below the registered `>30/76` alignment-routing criterion and increases negative activation; status is **PARTIAL / no alignment**.

### B84S-PROTO fixed prototype-anchor diagnostic

The final registered source-representation diagnostic selected by maximum
cosine to the fixed three contiguous causal source prototypes. At prefix16 it
reached `23/76` positive and `11/76` negative reliable selections, compared with raw source-mean `20/76` and `8/76`. This is below the `>30/76` alignment criterion and has higher negative activation than B84S-RA; it is **FAIL / no alignment**. Full event taxonomy is in `/data2/usr_for_deadline/trackocd_phase84/project_outputs/audit/b84sproto_failure_audit.json`.

### Event-level failure evidence

The repaired B84S-Q prefix16 taxonomy is retained in `/data2/usr_for_deadline/trackocd_phase84/project_outputs/audit/b84sq_failure_audit.json`. Its fold view is:

| event fold | polarity | events | selected | reliable | raw reliable | taxonomy |
|---|---|---|---|---|---|---|
| 0 | positive | 12 | 5 | 0 | 1 | {'candidate_selected_but_iou_unreliable': 2, 'defer_with_reliable_target_available': 6, 'source_observability_unreliable': 4} |
| 0 | negative | 12 | 4 | 0 | 0 | {'candidate_selected_but_iou_unreliable': 3, 'defer_with_reliable_target_available': 4, 'source_observability_unreliable': 4, 'target_observability_unreliable': 1} |
| 1 | positive | 12 | 2 | 0 | 3 | {'defer_with_reliable_target_available': 2, 'source_observability_unreliable': 8, 'target_observability_unreliable': 2} |
| 1 | negative | 12 | 2 | 0 | 1 | {'candidate_selected_but_iou_unreliable': 1, 'defer_with_reliable_target_available': 1, 'source_observability_unreliable': 6, 'target_observability_unreliable': 4} |
| 2 | positive | 24 | 8 | 2 | 8 | {'candidate_selected_but_iou_unreliable': 1, 'defer_with_reliable_target_available': 7, 'learned_selection_reliable': 2, 'source_observability_unreliable': 7, 'target_observability_unreliable': 7} |
| 2 | negative | 24 | 5 | 0 | 5 | {'candidate_selected_but_iou_unreliable': 1, 'defer_with_reliable_target_available': 6, 'source_observability_unreliable': 12, 'target_observability_unreliable': 5} |
| 3 | positive | 28 | 11 | 5 | 8 | {'defer_with_reliable_target_available': 3, 'learned_selection_reliable': 2, 'source_observability_unreliable': 8, 'target_observability_unreliable': 15} |
| 3 | negative | 28 | 8 | 2 | 2 | {'candidate_selected_but_iou_unreliable': 2, 'defer_with_reliable_target_available': 5, 'learned_selection_reliable': 1, 'source_observability_unreliable': 5, 'target_observability_unreliable': 15} |

The support-alignment callgraph confirms that B84S/B84S-Q/B84S-RA/B84S-PROTO implement selection only; no transformed support IoU is present. Alignment is therefore **NOT_RUN**, not zeroed.

## Route gates

| route | status | evidence |
|---|---|---|
| A84P true physical→R | FAIL | full native canonical membership, Q0 parity, unsafe/2-of-4 non-decreasing gate |
| B84S original | FAIL | query-agnostic source attachment; p16 reliable selection below comparator |
| B84S-Q repaired matcher | FAIL | 7/76 positive reliable at p16; repaired query contract still does not preserve raw signal |
| B84S-RA raw-anchor | PARTIAL | 24/76 positive, 9/76 negative; below >30/76 alignment criterion |
| B84S-PROTO fixed M=3 prototypes | FAIL | 23/76 positive, 11/76 negative; below alignment criterion |
| B84A alignment | NOT_RUN | selection criterion not met; no transformed-support implementation |
| C84 controller / Commit-CT | NOT_RUN | R/O routes did not authorize controller |
| sealed/public | NOT_RUN | sealed boundary remained closed |

## Resource, process, and repair audit

The run used bounded CPU workers for B84S/B84S-Q and no GPU training; no OOM or external-process termination occurred. GPU/RAM/disk snapshots and process state are in `/data2/usr_for_deadline/trackocd_phase84/project_outputs/audit/research_ledger.json`. Symlinked output storage is recorded in the registration and manifests; large native data/checkpoints remain on `/data2/usr_for_deadline/trackocd_phase84` or prior read-only Phase83 targets.

Uncompleted `.launched` markers are preserved as failed evidence (not relabeled): `b84s_b84s_smoke_r1_f0.launched, b84s_formal_r1.launched`. Repair records, commands, compile checks, hashes, and intentionally unrun historical suites are in `/data2/usr_for_deadline/trackocd_phase84/project_outputs/audit/repair_events.json` and `/data2/usr_for_deadline/trackocd_phase84/project_outputs/audit/validation_evidence_ledger.json`. No Phase84 process remained at finalization.

## Reproduction

```bash
cd /data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
/home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase84/run_full_temporal_physical.py --tag full_temporal_r1
/home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase84/build_physical_r_adapter.py
/home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase84/evaluate_b84s_event_replay.py --model-prefix b84sq_b84sq_formal_v3 --fold-count 3 --suffix _b84sq_v3 --manifest outputs/iclr27_phase84/manifests/b84sq_balanced_v3_manifest.json
/home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase84/evaluate_b84s_event_replay.py --model-prefix b84sq_b84sq_formal_v3 --fold-count 3 --suffix _b84sra_v1 --manifest outputs/iclr27_phase84/manifests/b84sq_balanced_v3_manifest.json --raw-anchor --raw-anchor-bound 0.05
```

The formal B84S-Q checkpoints and hashes are in the fold metrics and formal aggregate; the event replay is frozen and post-hoc. The machine-readable decision is `/data2/usr_for_deadline/trackocd_phase84/project_outputs/audit/phase84_decision.json` and report provenance is `/data2/usr_for_deadline/trackocd_phase84/project_outputs/audit/report_provenance.json`.

## Conclusion and next direction

Phase84 resolves the historical interface confounds and supplies valid negative/partial evidence: physical reassociation changed canonical membership but did not safely transfer to frozen R; the native candidate pool is present; the query-conditioned matcher still fails to generalize its ranking under the sparse disjoint TRAIN contract; a bounded raw anchor recovers a small amount but does not cross the registered alignment gate. The final MOT+OCD causal controller and sealed persistent Commit-CT remain unmeasured in this window. A future window should register one new evidence-backed query-conditioned representation/support contract (with explicit train/runtime candidate parity and broader legal source coverage) before any controller or backbone work; threshold and StateMemory tuning are not justified by Phase84.

## Artifact index

The live/research ledgers and all hashes are preserved in `outputs/iclr27_phase84/audit/`; the complete source list used for the headline tables is in `report_provenance.json`. Historical Phase83 outputs and reports were not overwritten.
