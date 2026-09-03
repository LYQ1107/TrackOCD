# TrackOCD Phase82P+ — Q0-Preserving Causal Physical Association

## Status

This report records the registered Phase82P window (`2026-09-03T16:39:00.669566Z` to `2026-09-04T02:39:00.669566Z`) and the work completed before the deadline. The route is a TRAIN-only residual fragment-repair experiment anchored to the frozen OVTR/Q0 physical stream. It is not a claim of MOT+OCD success.

Starting code was the Phase81P frozen head `319fc3b91e8bb93398c9c6f54569d4990ba066d0`; final code head after every coherent change was pushed to GitHub is `b7fed9397905f00db4648c511e54dfa1d2350b88` (`main`, origin synchronized).

## Frozen contract and boundaries

- Q0 detector/proposals/base score, physical lineage, five-field row keys, parent assignment, denominator and Phase75B causal evaluator were read-only.
- TRAIN fitting excluded all event videos. Four deterministic video-disjoint folds were used; category and track labels were metadata/target construction only.
- Inference tensors contain normalized box geometry, score/age/gap, causal velocity and a fixed projection of DINOv2 crop appearance. Category names, text, category/semantic/physical IDs, future rows/tracks, held GT, DEV+, Q1 and sealed/public-new labels are not inputs.
- Outputs and large features are under `/data2/usr_for_deadline/trackocd_phase82p` through the project symlink `outputs/iclr27_phase82p -> /data2/usr_for_deadline/trackocd_phase82p/project_outputs`.

## Q0 parity

The read-only Phase75B wrapper produced exact parity before any learned route: 152 event rows (76 positive and 76 negative), with positive both-reliable counts 17/22/22/23/25 at prefixes 1/2/4/8/16 and p16 source 49/76, target 40/76, both 25/76. No event labels were joined before inference. Artifact: `outputs/iclr27_phase82p/audit/strict_o_parity.json` (SHA256 `4582aa007d8aa0199cf98632105a1d1cc9315c80840ac35a1ee90237edd3336c`).

## Data and appearance preparation

The Q0 TRAIN stream contains 111,387 rows. A four-GPU DINOv2 ViT-B/14 crop extraction (GPUs 4–7) completed 27,847/27,847, 27,847/27,847, 27,847/27,847 and 27,846/27,846 rows with zero failures. The merged 111,387×768 float16 cache is `outputs/iclr27_phase82p/features/q0_dinov2.npz`, SHA256 `878451f2178f117c0919a3b2688bcb494077e1e63ea2ace3ab3b7b47163de902`. The failed first extraction (wrong `annotations/frames` root) remains in `outputs/iclr27_phase82p/logs/appearance/` and is not hidden.

The actual observation vector is 48-D: 8 box/center/size + 4 score/causal scalars + 4 velocity + 32 fixed DINOv2 projection dimensions. A schema assertion initially expected 49-D; the smallest repair changed only this constant, model input and config, with no row or label change.

## TRAIN residual manifest

The per-video builder resets history and maps each video's sorted observed frame indices to ordinal causal steps. This is necessary because Q0 TRAIN images are sampled at 30-raw-frame intervals; applying the 16-step horizon to raw frame numbers incorrectly produced zero candidates. The repaired manifest has 33,594 birth examples and 1,127 positive reconnect labels (positive counts per fit/validation fold are shown below), with no cross-video history:

| fold | fit examples | fit positive | validation examples | validation positive | validation mean history |
|---:|---:|---:|---:|---:|---:|
| 0 | 22,449 | 899 | 11,145 | 228 | 1.83 |
| 1 | 25,450 | 819 | 8,144 | 308 | 2.10 |
| 2 | 25,213 | 787 | 8,381 | 340 | 1.76 |
| 3 | 27,670 | 876 | 5,924 | 251 | 2.38 |

The fold counts sum to 1,127 positives in fit+validation for each fold because each validation partition is a held video slice of the same fixed examples. Manifest SHA256 is `3e56498eb363e81365d14778968fe2aa2410364ba89de7283a821849b1a8a081`.

## Model and training

The residual model is a two-layer, four-head temporal Transformer (`K=8`, hidden 256) over each candidate's causal history, with current-proposal and pair-geometry heads. Its listwise decision is `KEEP_Q0_BIRTH` or `RECONNECT`; the Q0 continuation stream is byte-preserved. The fixed asymmetric loss weights false reconnect 2× versus missed repair 1×. Checkpoints were saved at steps 1, 500 and 1,000.

| run | fold | steps | validation loss | validation accuracy | predicted reconnect | false reconnect | repair precision/recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| smoke | 0 | 2 | 2.3065 | 0.9795 | 0 | 0 | 0 / 0 |
| targeted | 0 | 500 | 0.1266 | 0.9795 | 0 | 0 | 0 / 0 |
| formal | 0 | 1,000 | 0.1229 | 0.9795 | 0 | 0 | 0 / 0 |
| formal | 1 | 1,000 | 0.2415 | 0.9622 | 0 | 0 | 0 / 0 |
| formal | 2 | 1,000 | 0.2684 | 0.9594 | 0 | 0 | 0 / 0 |
| formal | 3 | 1,000 | 0.2353 | 0.9576 | 0 | 0 | 0 / 0 |

The low loss and high accuracy are dominated by the KEEP class and do not demonstrate useful association. All formal units completed with atomic `.launched` and `.done` markers and valid `latest.pt`/`fold*_final.json` artifacts. Checkpoint hashes:

| fold | latest checkpoint SHA256 |
|---:|---|
| 0 | `82e49c6f6595aa90a2e171f2c3e95d1d4c1bd01d81c0dab96e352146663d4294` |
| 1 | `49630feebbc86d5fef145199b351f5ec8a6386fe507d3ea4615c9e1ce33ad105` |
| 2 | `000a23e9635818dce66743de6080957ae30b9acf5721a470def1168846162642` |
| 3 | `b9a61d9f069dc92b783198fd970d6a0f58b2f02ba28b0e5b5b8179a6f938aae6` |

## Failure and repair ledger

1. Initial appearance extraction used the wrong frame root. It failed on all four shards with `FileNotFoundError`; the root was corrected to `TAO-Amodal/frames`, then full extraction passed.
2. Manifest dimension assertion exposed 48-vs-49 schema drift. Constants/config/model were aligned to 48 and a per-video targeted assertion passed.
3. Raw-frame horizon produced zero TRAIN candidates because rows are sampled every 30 frames. Ordinal observed-step chronology restored causal candidates and 1,127 positive labels.
4. A tool-window return left two duplicate manifest builders (PIDs 15010 and 15854, descendants 15856/15857). Only those task-owned PIDs were explicitly SIGTERM-ed; old zero-candidate mtime/hash remained unchanged. The successful rebuild used one blocking session.
5. Direct entrypoints lacked the project root in `sys.path`; train/replay scripts were made self-contained and smoke passed.
6. Native lineage termination records have no bbox. The extractor/replay now skip those bookkeeping rows; a new 16-row native smoke passed with zero failures. The original failure is retained.

Full machine-readable details are in `outputs/iclr27_phase82p/audit/repair_events.json` and `validation_evidence_ledger.json`.

## Physical/O/R status

The frozen Q0 strict O reference remains 25/76. The residual formal run did not produce a reconnect decision, so it is expected to be byte-identical to Q0 on native physical lineage. Full 145,429-row native appearance extraction, residual replay, learned strict O, TrackEval learned physical metrics, Phase82P-S selective overwrite and Phase82P-F full association were not run: the native cache is a required prerequisite and its expected runtime exceeded the remaining registered window. The 16-row smoke is only an input-path check, not an O result. No controller, StateMemory, correspondence, threshold sweep or sealed/public evaluation was run.

Consequently no learned physical safety gate, O improvement, R gate or C gate is claimed. The residual route's actionable result is that its current TRAIN objective collapses to the conservative Q0 fallback despite nonempty causal candidates; a future run should first rebalance/inspect reconnect supervision and use the completed native cache path, then escalate to the already-authorized selective-overwrite route rather than altering Q0 or interpreting training loss as success.

## Resources and sealed status

At training preflight: 125 GiB RAM, 103 GiB available, no swap; GPUs 4–7 were idle and used one worker per GPU. GPU1 was an unrelated external process (never touched). No OOM, swap, or memory-pressure event occurred. `/data1` remained nearly full; large artifacts were written to `/data2` via symlink. No DEV+, Q1, public-new or sealed labels/files were accessed.

## Reproduction commands

```bash
cd /data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
/home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase82p/run_strict_o_parity.py
/home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase82p/build_residual_manifest.py --appearance outputs/iclr27_phase82p/features/q0_dinov2.npz
bash scripts/iclr27_phase82p/run_four_fold_residual_supervisor.sh residual_formal 1000
```

## Decision

`AUTONOMOUS_10H_COMPLETE_WITH_NEGATIVE_EVIDENCE` for the executed residual TRAIN route, not universal TrackOCD infeasibility. Q0 strict O parity and the conservative physical anchor remain valid; this residual route has no demonstrated repair. The next high-value action is to complete native appearance/replay in a fresh registered window, then run the pre-authorized selective overwrite/full joint-assignment route with exact TrackEval and strict O before considering any downstream semantic/controller work.
