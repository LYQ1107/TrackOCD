# TrackOCD Phase82R+ Autonomous Research Report

**Window:** 2026-09-04 02:17:53 UTC – registered deadline 12:17:53 UTC  
**Project:** `/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT`  
**Final code HEAD:** `054910c06fca98e412792135045836203ffadadc` (`origin/main` synchronized)  
**Status:** physical-association escalation completed with negative strict-O evidence; downstream retrieval/controller/sealed stages were not authorized by the frozen contract.

## Executive result

The frozen Q0/Phase75B causal contract was preserved.  Two TRAIN-only learned association routes and one causal temporal-appearance route were implemented, smoke-tested, trained/replayed where applicable, and compared on the complete 91-video native event stream (145,429 rows).  The best physical diagnostic was the temporal appearance mean: TRAIN class-agnostic TrackEval HOTA 14.755, AssA 32.825, IDF1 8.018 and IDSW 2,606 versus Q0 14.619/32.246/7.749/2,647.  This is a small physical-lineage improvement, not an OCD result.

The frozen positive observability result did not change for any route: **strict p16 O = 25/76** (source 49/76, target 40/76; fold both-reliable `[8, 2, 10, 5]`).  Therefore no retrieval R, causal StateMemory/controller C, Commit-CT, or sealed/public evaluation was run.  This follows the registered rule and avoids presenting physical or retrieval proxies as end-to-end OCD success.

## Frozen protocol and boundaries

- Q0 is the physical anchor.  Phase68 reference: top-20 IoU≥0.5 recall `71062/112798 = 0.629993`, macro HOTA `0.844035`; Phase82P strict-O parity remains 25/76 at p16.
- The native lineage contains 145,429 rows over 91 TRAIN videos, with 26,009 birth, 73,467 continuation and 45,953 termination records.  All rows, boxes, base scores and non-birth proposal fields are retained in every replay; only causal physical lineage parents may change.
- The positive denominator is 76, with 76 negative events retained by the evaluator.  The strict row rule is unchanged: `assigned == 1` and transformed IoU ≥ 0.5.  Prefix reference for Q0 is p1/p2/p4/p8/p16 = `17/76, 22/76, 22/76, 23/76, 25/76`.
- TRAIN fitting excludes all event videos.  The full-association manifest has 84,489 all-row examples across 43 non-event TRAIN videos; fold fit/validation existing-assignment counts are `(4773/1788), (4760/1801), (5143/1418), (5007/1554)`.
- Inference tensors contain normalized box/center/size, score/age/gap, causal velocity, fixed DINOv2 projection and causal history.  Category names, text, category/semantic/physical IDs, future rows/tracks, held GT, DEV+, Q1, public-new and sealed labels are not model inputs.  TRAIN labels are used only to construct targets and post-hoc audits.

## Phase82P baseline retained

The original residual route collapsed to Q0 KEEP (33,594 birth examples; 1,127 reconnect positives; four formal folds predicted no reconnect).  Phase82R first corrected the DINOv2 crop cache because TAO boxes are xywh, not xyxy.  Corrected Q0 cache SHA256 is `735bd5bf037666382f2995804825cc321c7f42a1c35389de33ca9bcec6601c0f`; native event cache SHA256 is `fecdfc3bf341fc28f81fed2fa19dba57063c49793a083f3d1c18835a9d722245`.  The correction was independently smoke-tested (distinct finite vectors) and did not alter Q0 rows.

## Route 1 — full learned causal association

`FullAssociation` is a separate one-layer GRU over K=8 causal candidate histories, with a masked 16-way existing-candidate head and an explicit NEW head.  Training uses balanced existing/NEW batches and masked cross-entropy; no controller, StateMemory or semantic action is present.

| fold | updates | val existing precision | val existing recall | val pred-existing |
|---:|---:|---:|---:|---:|
| 0 | 6,315 | 0.3226 | 0.3714 | 0.0796 |
| 1 | 6,825 | 0.1642 | 0.3454 | 0.1756 |
| 2 | 7,110 | 0.2992 | 0.2764 | 0.0700 |
| 3 | 7,170 | 0.2914 | 0.3752 | 0.1090 |

Fold 0 was selected solely by TRAIN validation existing-F1.  Its native replay made 509 reconnect decisions (25,500 KEEP) with 145,429 rows.  Physical proxy: 1,023 unique tracks, 2,758 GT switches, 353 fragmented GT tracks, 372 merged tracks and 1,858 duplicate-birth proxy.  TRAIN TrackEval summary: HOTA 14.647, DetA 6.8156, AssA 32.324, MOTA −822.29, IDF1 7.8376, IDSW 2,644, Frag 747.  Strict O remained 25/76.

## Route 2 — explicit raw-appearance anchor

The only registered model variant added an explicit cosine between current and candidate DINOv2 projection while preserving the same data, seed, candidate pool, causal action and loss.  Four-fold validation existing recall was 0.4044/0.3604/0.2927/0.4022.  Fold 0 replay made 620 reconnects; replaying all four fold checkpoints (diagnostic only, no held-event selection) made 620/542/434/448 reconnects, and every strict-O result remained 25/76.

Selected fold 0 physical proxy: 1,024 tracks, 2,759 GT switches, 353 fragmented, 366 merged, duplicate-birth proxy 1,852.  TrackEval: HOTA 14.707, DetA 6.8178, AssA 32.598, MOTA −822.30, IDF1 7.8589, IDSW 2,645, Frag 747.  This is non-inferior physical diagnostics but no O improvement.

## Route 3 — causal temporal appearance mean

Because the native raw appearance score had measurable TRAIN signal, a parameter-free Level-4 route was registered: candidate appearance is the normalized running sum of already observed vectors, rather than the last vector.  Motion/geometry weights, dormant-only candidate eligibility, acceptance score 0.5, collision fallback and row preservation are unchanged.

The one-video smoke produced 2,069 rows and 74 reconnects.  Full replay produced 8,035 reconnects, 797 collision groups (all handled by explicit collision fallback), 20,468 canonical state merges and no row/box deletion.  Physical proxy improved relative to Q0: 964 tracks, 2,723 GT switches, 353 fragmented, 327 merged, duplicate-birth proxy 1,731.  TRAIN TrackEval was:

| stream | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | Frag |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q0 native | 14.619 | 6.8149 | 32.246 | −822.32 | 7.7485 | 2,647 | 748 |
| full causal r3 | 14.732 | 6.8146 | 32.724 | −821.94 | 7.9595 | 2,610 | 748 |
| learned full association | 14.647 | 6.8156 | 32.324 | −822.29 | 7.8376 | 2,644 | 747 |
| raw-anchor association | 14.707 | 6.8178 | 32.598 | −822.30 | 7.8589 | 2,645 | 747 |
| temporal appearance mean | **14.755** | 6.8125 | **32.825** | **−821.90** | **8.0176** | **2,606** | 748 |

These are TRAIN-only class-agnostic TrackEval diagnostics on the native Q0 stream, not the Phase68 validation anchor and not an OCD score.

## Strict O comparison

| route | p16 source | p16 target | p16 both reliable | fold both-reliable |
|---|---:|---:|---:|---|
| Q0 / Phase82P parity | 49/76 | 40/76 | **25/76** | `[8,2,10,5]` |
| full learned association | 49/76 | 40/76 | **25/76** | `[8,2,10,5]` |
| raw-anchor association | 49/76 | 40/76 | **25/76** | `[8,2,10,5]` |
| temporal appearance mean | 49/76 | 40/76 | **25/76** | `[8,2,10,5]` |

All four fold-checkpoint replays of the raw-anchor route gave the same strict-O vector.  The unchanged O ceiling is expected to be dominated by fixed proposal/support observability; it is not evidence that the TRAIN association loss has zero signal.

## Failure/repair and resource ledger

1. The first all-row manifest invocation failed before writing scientific output because the direct entrypoint lacked the project root in `sys.path`.  A minimal path fix was committed, then one-video smoke passed.
2. The second smoke exposed a missing `frame` key expected by the read-only Phase82P observation helper.  Adding causal `frame`/`age` state metadata was the minimal fix; smoke and full rebuild passed.  No Phase82P file was changed.
3. The initial full-association supervisor used system Python without PyTorch.  It exited before training; the supervisor was changed to the audited `/home/lwr/anaconda3/envs/ovtr/bin/python` runtime before formal workers were started.
4. A raw-anchor TRAIN validation sweep was explicitly SIGTERM-ed after roughly ten minutes (task-owned PIDs 31552 and 31553) because it duplicated the registered replay diagnostic.  No external process was touched; this interruption is retained as evidence and is not a model result.
5. No OOM, swap, duplicate formal supervisor or GPU collision occurred.  Formal workers used one GPU each on GPUs 4–7; GPU0–3 were left untouched when occupied by unrelated work.  At final audit RAM available was about 99 GB of 125 GB; `/data2` held approximately 3.9 GB of Phase82R outputs/data.  Large arrays/checkpoints are under `/data2/usr_for_deadline/trackocd_phase82r` and surfaced through the symlink `outputs/iclr27_phase82r`.

Code and log changes were pushed before this report.  The main commits are `471deea` (model/config), `0edfb8f` (manifest), `959d72d`/`d4ab7ca`/`2b3fe8e` (entrypoint fixes), `ae04dbe`/`4acda87` (training/supervisor), `78c10b8` (replay), `55218dc` (raw-anchor), `8f769cd` (temporal mean), and `054910c` (log).  `origin/main` equals final HEAD `054910c`.

## Gate and downstream status

- **Physical diagnostic:** non-inferior; temporal mean is the best diagnostic stream, but its lower track count and collision handling are reported rather than hidden.
- **Strict O improvement gate:** **FAIL** (25/76 → 25/76, zero delta).
- **Retrieval R:** **NOT_RUN** under the registered Phase82R rule; no learned retrieval claim is made.
- **Causal controller / StateMemory / Commit-CT:** **NOT_RUN**; there is no valid Phase82R Commit-CT number to report.
- **Sealed/public/Q1:** **NOT ACCESSED**.

The evidence supports a narrow conclusion: corrected DINOv2 has usable TRAIN association signal, and causal temporal aggregation can improve physical-lineage diagnostics, but fixed proposal observability and event-support coverage remain unchanged.  Neither learned association route is authorized to masquerade as end-to-end OCD.  A future registered phase should address proposal/support coverage or construct legally aligned fragment-transition supervision before attempting retrieval/controller evaluation; threshold, StateMemory, controller and backbone lottery are not justified by this window.

## Reproduction

```bash
cd /data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
python scripts/iclr27_phase82r/build_full_association_manifest.py \
  --appearance outputs/iclr27_phase82r/features/q0_dinov2_corrected_r1.npz
bash scripts/iclr27_phase82r/run_full_association_supervisor.sh full_assoc_formal_r1 15
bash scripts/iclr27_phase82r/run_full_association_supervisor.sh full_assoc_raw_anchor_formal_r1 15 raw_anchor
/home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase82r/replay_full_association.py \
  --checkpoint outputs/iclr27_phase82r/checkpoints/full_assoc_raw_anchor_formal_r1/fold0/latest.pt \
  --tag full_assoc_raw_anchor_replay_r1 --explicit-app-cosine
/home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase82r/replay_full_causal_assignment.py \
  --tag temporal_app_mean_r1 --temporal-app-mean
/home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase82r/evaluate_strict_o.py \
  --replay outputs/iclr27_phase82r/replays/temporal_app_mean_r1.jsonl \
  --tag temporal_app_mean_r1
```

Machine-readable decision: `outputs/iclr27_phase82r/audit/phase82r_decision.json`.  Evidence ledger: `outputs/iclr27_phase82r/audit/validation_evidence_ledger.json`.
