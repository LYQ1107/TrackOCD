# TrackOCD Phase80+ — Autonomous 10-hour research report

**Final status:** `AUTONOMOUS_EARLY_STOP_NO_COMPLIANT_ROUTE`  
**Project:** `/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT`  
**Session:** Luna `01a01fb6-96f7-7132-a318-0833180c88d8`  
**Research start (UTC):** `2026-09-02T18:27:32Z`  
**Registered deadline (UTC):** `2026-09-03T04:27:32Z`  
**Actual end (UTC):** `2026-09-02T20:17:27Z`  
**Actual runtime:** approximately `1.83 h`  
**Ending HEAD:** `6db4a3181f025b050014d4cd4b45a784c3ef5454`

This report is the Phase80+ evidence ledger.  It is intentionally kept
separate from the historical Phase15S–79 reports.  The window ended early only
because the registered A/B hypotheses failed and the remaining C/D references
had no compliant, reproducible execution path under the frozen causal
contract; no unregistered long training was started.  This is not a claim that
TrackOCD as a research problem is solved or impossible.

## 1. Research question and frozen boundary

The Phase80+ family asks whether a new visual representation source or a
causal-memory-matched scorer can improve cross-physical-track/cross-video
correspondence without breaking the online TrackOCD contract.  It does **not**
change the physical tracker, StateMemory, controller, thresholds, event
denominator or sealed protocol.

The inference boundary was frozen to visual/geometry/history information.  No
category name, text vocabulary, semantic ID, physical ID feature, future row or
track, held label, DEV+, Q1, public-new or sealed label was provided to a model.
TRAIN category/track fields were used only as supervision metadata.  No
controller, Commit-CT or sealed evaluation was run in Phase80 because the
registered representation gates did not pass.

The explicit no-rerun list was Phase75C/D/E, Phase76A/AR/S/G/X and Phase79O
unless an algorithm changed.  Their frozen results remain historical evidence.

## 2. Historical audit and external methods

The read-only history audit found that the existing DINOv3 bakeoff used a timm
conversion and global/track-level CLS or mean features; it did not measure the
Phase80 dense patch source.  Its recorded status was `NO_CLEAR_DINOV3_GAIN`.
The local weight-integrity record has 85,641,216 parameters, feature dimension
768, four register tokens, 16-pixel patches and SHA-256
`1f9ed8a2378d65e24bb710ba522ac9fa7be4e036d7aefb4384ce022833926332`, with 18/18
integrity checks passing.  Phase80 therefore labeled the new timm checkpoint
`TIMM_DISTRIBUTION`, not Meta-official DINOv3.

Official references checked during the family pivots were:

| reference | official source and revision | usable boundary |
|---|---|---|
| DINOv3 | [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3), main `6876159a11b4df116f30f667f8c9888617df0751`; [paper](https://arxiv.org/abs/2508.10104); DINOv3 License | dense/sparse patch matching is relevant, but the local Phase80 run used an already cached timm distribution and did not claim official equivalence |
| Grounded Correspondence | [ICML2026-RethinkingOCL](https://github.com/LiZhYun/ICML2026-RethinkingOCL), `5d345268797425558b449337519af3ab24aeb6f1`; [paper](https://arxiv.org/abs/2605.03650) | audited as a correspondence reference; no code/weights imported |
| TRACT | [Nathan-Li123/TRACT](https://github.com/Nathan-Li123/TRACT), `19f01d72f9f6c212c28fd9cb0171a5432cd41a6a`; [ICCV 2025 paper](https://openaccess.thecvf.com/content/ICCV2025/html/Li_TRACT_ICCV2025_paper.html) | trajectory aggregation is relevant, but the release expects external proposals/MASA and exposes TraCLIP text/category paths |
| ObjectRelator | [insait-institute/ObjectRelator](https://github.com/insait-institute/ObjectRelator), `25ecbc086cc812304de97764aa21f4bb8e0e6360`; [ICCV 2025 paper](https://openaccess.thecvf.com/content/ICCV2025/html/Qian_ObjectRelator_ICCV2025_paper.html) | static cross-view correspondence reference, not causal persistent MOT |
| TrajViT | [RAIVNLab/trajvit](https://github.com/RAIVNLab/trajvit), `8fe9949dd86435bebc2c35d8b23d77a019c487a2`; [paper](https://arxiv.org/abs/2505.23617) | trajectory tokens, but official pretraining assumes eight GPUs, external SAM2 trajectories and caption metadata; no exposed license file |
| Trace Anything | [ByteDance-Seed/TraceAnything](https://github.com/ByteDance-Seed/TraceAnything), `54677b5e7bf11510c2e8c917a509988ad379f8eb`; [paper](https://arxiv.org/abs/2510.13802) | 4D trajectory fields (Apache-2.0 code, CC BY-NC weights), not semantic track correspondence or Commit/Defer |

No external model was downloaded.  Method boundaries are recorded in
`outputs/iclr27_phase80d/audit/{tract_route_audit,modern_trajectory_audit}.json`.

## 3. Family A — dense visual source

### Implementation

The new source extracts a deterministic 10% context crop from each frozen
track row at 256×256 with ImageNet normalization.  Timm ViT-B/16 DINOv3
features have shape `[CLS, 4 register, 256 patch, 768]`; registers were
excluded and a fixed 4×8 (32-token) patch subset was retained alongside the
normalized CLS.  Edge/narrow boxes receive a deterministic minimum four-pixel
crop; row keys and the denominator are unchanged.  Four shards were extracted
on GPUs 4–7 and stored on `/data2` through the project cache symlink.

The first extraction attempt failed before writing shards 1–3 because 22
edge-touching boxes were narrower than the crop guard (1/12/9 rows per shard).
The minimal crop expansion fixed only those rows.  The first diagnostic counted
988 queries because four queries had no positive or negative candidate; the
minimal evaluator fix retained them in an `unevaluable` list and excluded them,
matching the frozen Phase75D scorer's 984-query contract.

### Frozen diagnostic (same Phase30 TRAIN-disjoint protocol)

At prefix16 the exact raw parity check passed:

| source | R@1 | mAP | hard-gap | rescued raw-wrong | harmed raw-correct | net rescue |
|---|---:|---:|---:|---:|---:|---:|
| old DINOv2 raw | 0.893219 | 0.848374 | 0.189559 | — | — | — |
| DINOv3 global CLS | 0.891365 | 0.838037 | 0.193225 | 12 | 9 | +3 |
| DINOv3 dense patch relation | 0.835144 | 0.822818 | 0.087886 | 9 | 17 | −8 |

Dense per-fold prefix16 net rescues were `[+1, −3, −2, −4]`; three folds had
R@1 drops below −0.02.  The registered routing condition (more rescue than
harm, at least three positive folds and no catastrophic drop) failed.  No
router or controller was trained.  Full prefix/fold rows remain in
`outputs/iclr27_phase80a/metrics/phase80a_dense_diagnostic.json`.

**Decision:** close the dense-source sub-route and pivot to Family B.

## 4. Family B — causal-memory-matched supervision

### Contract and model

The model is a small sequential candidate-set evidence scorer.  It consumes
only the causal raw cosine sequence, prefix delta, candidate rank, entropy, age
and its own previous evidence state; it outputs a bounded residual on the raw
score.  Losses were listwise ranking, hard-negative weight 0.35, prefix
persistence 0.5, raw safety 1.5 and residual 0.02.  The read-only Phase76AR
`memory_mimic` banks were video/category-disjoint by fold.

The audit showed a large fit→validation distribution shift at prefix16 in raw
top-1 positive rate: f0 `0.4839→0.2485`, f1 `0.0779→0.9390`, f2
`0.0448→0.5946`, f3 `0.0512→0.6429`.  This was registered before training and
was not tuned with held outcomes.

### Execution and repair ledger

- 100-step smoke: finite state, exact raw fallback at initialization, unsafe 0.
- 500-step fold0 targeted: R@1 unchanged; mAP `+0.000260`; hard-gap
  `+0.000182`; unsafe 0.
- Four formal workers ran once on GPUs 4–7 (5,000-update target each; best
  steps f0=500, f1=2,000, f2=5,000, f3=1,000). No OOM occurred.
- The first exact replay failed before writing metrics because the pinned torch
  rejected `weights_only`; a try/except legacy local-checkpoint fallback was
  the minimal repair.  A four-row fold0 load regression and the exact replay
  then passed.

### Exact TRAIN-disjoint replay (984 validation queries, prefix16)

| scope | raw R@1 | learned R@1 | ΔR@1 | raw mAP | learned mAP | ΔmAP | raw hard-gap | learned hard-gap | unsafe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aggregate | 0.606246 | 0.581414 | −0.024832 | 0.641163 | 0.594240 | −0.046923 | 0.046725 | 0.032875 | 8 |
| fold 0 | 0.248507 | 0.248507 | 0.000000 | 0.450136 | 0.450396 | +0.000260 | −0.065607 | −0.065426 | 0 |
| fold 1 | 0.939024 | 0.902439 | −0.036585 | 0.898666 | 0.817980 | −0.080686 | 0.203842 | 0.157969 | 3 |
| fold 2 | 0.594595 | 0.567568 | −0.027027 | 0.627964 | 0.570470 | −0.057493 | 0.011150 | −0.002444 | 2 |
| fold 3 | 0.642857 | 0.607143 | −0.035714 | 0.587884 | 0.538112 | −0.049772 | 0.037516 | 0.041401 | 3 |

The formal and exact results fail the registered R gate: aggregate R@1, mAP
and hard-gap decrease, three folds lose both R@1 and mAP, and unsafe flips are
non-zero.  The stateful model was not connected to StateMemory/controller.

**Decision:** `PHASE80B_GATE_R_FAIL_ROUTE_FAMILY_C_OR_D`.

## 5. Family C — proposal/observability audit

The evaluator-only audit reads the frozen Phase75B Q0 trace (760 prefix rows,
76 positive events).  At prefix16, a Q0 candidate with IoU ≥0.5 exists on 72/76
source sides and 64/76 target sides, but strict event reliability is 49/76,
40/76 and both sides 25/76.  Candidate-present assignment/temporal gaps are
23 source and 24 target.  The joint breakdown is:

| condition | events |
|---|---:|
| both event-reliable | 25 |
| both pool-good but assignment/temporal gap | 36 |
| source pool-good, target pool-insufficient | 11 |
| target pool-good, source pool-insufficient | 3 |
| pool-insufficient on both sides | 1 |

Per-fold both-reliable counts are `[8/12, 2/12, 10/24, 5/28]`.  Candidate-
present source gaps average temporal IoU 0.1699 with 11.96 fragmentation
transitions; target gaps average temporal IoU 0.1045 with 5.00 transitions.
Event-reliable source/target sides average 0.4863/0.6206 temporal IoU.  This is
strong evidence for physical-assignment/lifecycle headroom, not a safe
representation or controller result.  Phase79O's causal velocity projection
already retained raw candidates and left the prefix16 ceiling at 25/76, so no
quantity-expansion rerun was performed.

**Decision:** `PHASE80C_AUDIT_ASSIGNMENT_HEADROOM_NO_NEW_PHASE80_MODEL`.

## 6. Family D — modern trajectory references

TRACT, TrajViT and Trace Anything were checked against the causal TrackOCD
contract.  TRACT cannot be used without external proposals and its TraCLIP
scripts construct CLIP text/class-name cues.  TrajViT's public pretraining
assumes eight GPUs, external SAM2 trajectories and caption metadata, with no
exposed repository license.  Trace Anything provides scene-level 4D fields,
not semantic track embeddings; its model weights are CC BY-NC 4.0 and examples
require at least 48 GB VRAM.  None is a direct, licensed, four-GPU,
category-free prior-support correspondence module.  No weights were downloaded
or executed.

**Decision:** `REJECT_AS_PRIMARY_PHASE80_ROUTE` (audit-only; trajectory ideas
remain a future physical-association reference).

## 7. Resources, process and storage ledger

- Preflight: 125 GiB RAM, about 101–103 GiB available; GPUs 0/1 were occupied by
  external jobs and were not touched; GPUs 2–9 were available.  `/data1` had
  about 34 GiB free; `/data2` about 1.2 TiB free.
- All Phase80 workers were bounded to four and mapped one fold/shard per GPU on
  GPUs 4–7.  No OOM or external process intervention occurred.
- One duplicate Family-B audit was accidentally spawned after a tool return.
  Only task-owned PID 3170 and parent shell 3167 received explicit SIGTERM;
  original PID 2322 completed and atomically wrote the audit JSON.  No external
  process was touched.
- Dense cache: `/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/outputs/iclr27_phase80a/cache`
  → `/data2/usr_for_deadline/trackocd_phase80a/dense_cache` (valid symlink,
  about 1.9 GiB).  Family-B checkpoints are on
  `/data2/usr_for_deadline/trackocd_phase80b/checkpoints` and exposed as
  output symlinks (about 5.5 MiB).

## 8. Code and reproducibility

All substantive tracked code changes were compiled and pushed to
`https://github.com/LYQ1107/TrackOCD.git` on `main` before this report:

`eac6654`, `9ac7bd9`, `64972ba`, `8c7a94e`, `bb32667`, `a985aaf`, `676bb9a`,
`374cb00`, `48b80ad`, `491044b`, `fcd4488`, `39ea894`.

The latest pushed HEAD is recorded by
`outputs/iclr27_phase80/validation_evidence_ledger.json`.  Changed Python files
passed `py_compile`; shell supervisors passed `bash -n`.  Primary commands:

```text
bash scripts/iclr27_phase80a/run_dense_extract_supervisor.sh
PYTHONPATH=. /home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase80a/run_dense_diagnostic.py
bash scripts/iclr27_phase80b/run_four_fold_supervisor.sh phase80b_formal 5000
PYTHONPATH=. /home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase80b/evaluate_memory.py --tag phase80b_formal --run-id phase80b-exact-20260902-r1
PYTHONPATH=. /home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase80c/run_observability_audit.py
PYTHONPATH=. /home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase80d/audit_tract_route.py
PYTHONPATH=. /home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase80d/audit_modern_trajectory.py
```

Machine-readable ledger: `outputs/iclr27_phase80/validation_evidence_ledger.json`.
Phase80A/B/C/D reports and artifacts are retained under their respective
`docs/iclr27_phase80*` and `outputs/iclr27_phase80*` paths.  No public/Q1/DEV+
or sealed output was created.

## 9. Current scientific conclusion and unverified items

The new dense source failed the registered rescue routing criterion.  The
causal-memory scorer failed the strict safe multi-fold R gate.  The proposal
audit indicates that many positive events have visually close candidate boxes
but lose legal causal reliability through physical-track temporal assignment;
the modern trajectory references do not supply a compliant drop-in interface.
Consequently, this window has not authorized a controller, Commit-CT, or sealed
evaluation.  Retrieval gains, candidate-pool upper bounds and evaluator-only
IoU ceilings are not TrackOCD success.

Unverified by design in this window: full persistent controller behavior using a
new representation, standard full-sequence TrackEval for a new physical stream,
and sealed/public performance.  They remain blocked until a future authorized
route supplies a legal and stable physical-assignment/support contract.

The evidence-backed next direction is one separately authorized causal
physical-association and trajectory-completeness route (frame-level
visual/motion supervision with raw Q0 distillation and no semantic shortcut),
followed only if its observability improves by a frozen
correspondence/controller compatibility test.  Repeating the frozen-feature
memory/ranker family, threshold tuning, or modern-backbone lottery is not
justified by this ledger.

## Phase81P+ continuation (2026-09-03)

The Q0-anchored physical association window ran from `2026-09-03T08:40:36Z` with registered deadline `2026-09-03T18:40:36Z`. Four bounded GPU workers trained the initial association transformer and two evidence-based repairs (resolution-aware geometry and top-4 candidate-conditioned NEW context) on legal TRAIN-only Q0 proposal streams. Event observability rose from Q0 `25/76` to model means up to `60/76`, but this proxy gain did not survive physical safety: Q0 native event-stream proxies are 1,026 tracks/2,766 switches/354 fragmented GT tracks/383 merges/1,902 duplicate births, while the best learned route still averages 2,714 tracks/7,250 switches/403 fragmented tracks/1,028 merges/5,606 duplicate births. Therefore physical safety failed for all three versions; no controller or sealed evaluation was run. Source code was pushed through `319fc3b91e8bb93398c9c6f54569d4990ba066d0`; full evidence is in [`PHASE81P_CAUSAL_PHYSICAL_ASSOCIATION_REPORT.md`](docs/iclr27_phase81p/PHASE81P_CAUSAL_PHYSICAL_ASSOCIATION_REPORT.md) and `outputs/iclr27_phase81p/audit/validation_evidence_ledger.json`.

## Phase83 — dual-path physical→R and O-support (2026-09-04)

- Phase83 was registered from commit `596c8ff9193c4211468966985a7d8a9e738c8989` with Q0/Phase75B rows, 76+76 event denominator, prefixes and evaluator frozen. Outputs use the `/data2/usr_for_deadline/trackocd_phase83` symlink; no DEV+/Q1/public-new/sealed labels were accessed.
- Branch A mapped the Phase82R native temporal-appearance-mean lineage to corrected public rows. Only 1,046/6,213 tracks had usable matches (74/76 event pairs). Exact TRAIN validation p16 R@1 was 0.882735 vs raw 0.893219, mAP 0.847251 vs 0.848374, hard-gap 0.198022 vs 0.189559, with 5 unsafe flips; only one fold was non-decreasing in both R@1/mAP. Event-pair temporal cosine (0.197574) was below raw (0.271772). Decision: `R83_DIAGNOSTIC_NO_SAFE_IMPROVEMENT`; no downstream C run.
- Branch B read-only taxonomy reproduced the proposal-pool upper bound source/target/both 72/64/61 and frozen event reliability 49/40/25. Main p16 classes were B=15, D=18, E=36, G=7; no missing proposals. The TRAIN-only hidden-128 router completed 100-step smoke, 500-step fold0 targeted and 4×1000-step formal runs with atomic checkpoints. Learned p16 both-side reliable support was 8/76 (frozen 25/76); support was selected on 46/76 positives and 52/76 negatives. Decision: `O83_FAIL_NO_SUPPORT_GAIN`; no controller/StateMemory/threshold/backbone/sealed path was run.
- All code changes were committed/pushed before execution. No OOM or external-process termination occurred; GPU 0–3 remained external and untouched, GPU4–7 were not needed. Primary artifacts: `outputs/iclr27_phase83/audit/{support_assignment_callgraph,failure_taxonomy_76,failure_taxonomy_summary,observability_by_prefix}.json`, `outputs/iclr27_phase83/metrics/{physical_r_temporal,o_support_replay_formal}.json`, and `docs/iclr27_phase83/{PHYSICAL_TO_R_REPORT,O_SUPPORT_REPORT}.md`.
- Phase83 closes this window's two registered diagnostics without claiming OCD success. Persistent Commit-CT, unchanged-controller C83 and sealed evaluation remain `NOT_RUN`; a later route must repair the support/assignment interface or provide a separately registered legal proposal source rather than tune thresholds or declare retrieval/O oracle success.

