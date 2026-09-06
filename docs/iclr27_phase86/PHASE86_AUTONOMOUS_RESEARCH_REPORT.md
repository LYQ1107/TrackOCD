# TrackOCD Phase86 — Autonomous Research Report (final window snapshot)

**Status:** `PHASE86_WINDOW_COMPLETE_WITH_VALID_NEGATIVE_EVIDENCE`  
**Generated (UTC):** `2026-09-06T17:12:11.694531+00:00`  
**Original window:** `2026-09-06T07:57:11.490953Z` → `2026-09-06T17:57:11.490953Z`  
**Remaining at generation:** `2700s`  

> The earlier report was an interim premature snapshot. This report was generated only after the registered deadline-minus-45-minute unlock.

## 1. Execution repair and provenance

- Original registration was not rerun. Resume record: `/data2/usr_for_deadline/trackocd_phase86/project_outputs/audit/resume_after_premature_finalization.json`.
- The finalization guard rejects early generation and allows only the registered unlock interval or an explicit allowed hard blocker. No hard blocker was declared.
- Code/contract repair head before final report: `cc0956d9c6f4f4eb46f9a9e0c07ec2ee14b6efb3`; original premature report head: `e9801f9ca34fd135a02813b5709ae3c78d3ea8e7`.
- Old D0/D1/U1/RF artifacts remain read-only evidence. The old modulo-3 event replay is labeled `U1_CV_CHECKPOINT_ROUTING_EVENT_DIAGNOSTIC`; the old within-track chunks are `WITHIN_TRACK_4FRAME_CHUNK_DIAGNOSTIC_NEGATIVE`.
- GPU0 PID 33785 and external intermot workers were not touched. CPU routes used bounded single-process execution; no OOM or broad kill occurred. `/data1` was nearly full, so large outputs/checkpoints remained on `/data2` via the existing symlink.

## 2. Frozen boundaries

- No DEV+, Q1, public-new or sealed labels were used for training, checkpoint selection or inference. No future rows/tracks, category text, semantic IDs or physical IDs entered model tensors.
- The Phase19R controller/StateMemory, event denominator (76 positive + 76 negative), prefixes `{1,2,4,8,16}`, candidate metadata order and frozen manifests were not changed.
- Formal controller, Commit-CT and sealed evaluation were not run because no corrected score route passed its upstream safety gate.

## 3. Frozen D0/D1 diagnostic OCD

| stream | Commit-CT | existing F1 macro | negative false-merge | unresolved | premature | duplicate births | category/video coverage |
|---|---:|---:|---:|---:|---:|---:|---|
| D0 historical Q0 + frozen RC-MS-OCD | 3/76 | 0.0268 | 0.2842 | 0.4449 | 0.2664 | 84 | category=1, video=2 |
| D1 temporal physical + frozen RC-MS-OCD | 3/76 | 0.0268 | 0.2842 | 0.4449 | 0.2664 | 87 | category=1, video=2 |

D0 and D1 both remain diagnostic-only 3/76, all three correct events in fold3. The full metric artifact is `audit/diagnostic_ocd_full_metrics.json`; no D0/D1 replay was repeated during resume.

## 4. Corrected all-TRAIN U1 deployment

| route | p16 positive | p16 negative | reranker-use positive | reranker-use negative | decision |
|---|---:|---:|---:|---:|---|
| corrected all-TRAIN expert + all-OOF gate | 20/76 | 12/76 | 66/76 | 71/76 | **U1_FINAL_EVENT_SAFETY_FAIL** |

The support expert used the validated 19-D/10-D `SupportReranker`, 15 effective epochs, 54,780 steps and seed 86001. The gate used the unchanged 64→32 utility architecture, 2,000 steps and seed 86100. The two checkpoint paths and hashes are embedded in the replay artifact. Model selection never used event labels; event fold is reporting only, not model routing. The registered safety rule (positive≥22/76 and negative≤8/76) was not met.

## 5. Canonical-root RF

- Source lineage: `/data2/usr_for_deadline/trackocd_phase85/project_outputs/physical/temporal_mean_full/full_temporal_lineage.jsonl`; union timeline: `/data2/usr_for_deadline/trackocd_phase85/project_outputs/physical/temporal_mean_full/union_events.jsonl`.
- Reconstructed comparisons: `3527920`; fraction roots with >1 fragment: `0.2464`; maximum fragments: `6`; fallback comparisons: `830`.
- p16 raw R@1/mAP `0.893219/0.848374`; canonical-root `0.893219/0.849139`; unsafe flips `1`; non-decreasing folds `2/4`.
- Decision: `RF_CANONICAL_ROOT_V1_NEGATIVE`. It is a valid canonical-root test, not the old within-track diagnostic, and it does not authorize the controller.

## 6. U2 and recent-method audit

- U2-v1 three-fold TRAIN validation produced zero rescue and zero harm in every fold; raw top-1/top-5 were unchanged and the learned residual saturated to a constant. Decision: `U2_BALANCED_TRAIN_NEGATIVE`.
- One minimal repair balanced match/defer exposure without changing architecture, candidates, seed or protocol. It also produced zero rescue/harm in all folds. No third U2 variant was launched.
- The verified recent audit covers AGE, TALON, LTC, TRACT, COVTrack, ObjectRelator and C3Po with current remote heads and license status. None exposes a drop-in text-free causal prior-support selector for this schema, so no external code/weights were imported. See `docs/iclr27_phase86/RECENT_METHOD_AUDIT.md`.

## 7. Gate/status table

| gate | result | interpretation |
|---|---|---|
| frozen D0/D1 | diagnostic 3/76 | controller baseline evidence only |
| corrected U1 event safety | FAIL | 20/76 positive, 12/76 negative |
| canonical-root RF | FAIL | 2/4 non-decreasing and one unsafe flip |
| U2 TRAIN | FAIL | 0/3 folds with net rescue |
| controller/formal OCD | NOT RUN | no legal safe upstream stream |
| sealed/public | NOT RUN | sealed boundary preserved |

## 8. Reproduction

```bash
cd /data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
/home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase86/train_support_expert_final.py
/home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase86/train_u1_final.py
/home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase86/run_u1_final_event_replay.py --tag u1_final_alltrain_event_v3
/home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase86/run_rf_canonical_root.py --tag rf_canonical_root_v2
/home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase86/train_u2.py --mode cv --fold 0 --epochs 15 --steps 1000
```

## 9. Final research decision

**Phase86 did not complete MOT+OCD or sealed evaluation.** The resume corrected the deployment and execution contracts and exhausted the registered U1, canonical-root RF, U2-v1 and one justified U2 balancing repair. The evidence supports a narrower conclusion: current Phase85 support supervision/selector interface does not yield a safe upstream candidate under the frozen protocol. It is not evidence that the entire TrackOCD task is universally impossible. A future phase must register a new legal support contract or materially improve supervision; it must not repeat fold-modulo deployment, within-track RF chunks, or another gate/threshold lottery.

## 10. Artifact provenance

- Decision: `/data2/usr_for_deadline/trackocd_phase86/project_outputs/audit/phase86_decision.json`
- Ledger: `/data2/usr_for_deadline/trackocd_phase86/project_outputs/audit/research_ledger.json`
- U1 replay: `/data2/usr_for_deadline/trackocd_phase86/project_outputs/metrics/u1_final_alltrain_event_v3.json`
- RF replay: `/data2/usr_for_deadline/trackocd_phase86/project_outputs/metrics/rf_canonical_root_v2.json`
- U2 decision: `/data2/usr_for_deadline/trackocd_phase86/project_outputs/audit/u2_decision.json`
- Source code head: `cc0956d9c6f4f4eb46f9a9e0c07ec2ee14b6efb3`
- `public_dev_q1_sealed_accessed=false`, `formal_ocd_run=false`, `sealed_run=false`.
