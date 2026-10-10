# Train-first main protocol — T0 preregistration

Status: **ACTIVE_CORE_TRAINING_WITH_LIMITED_FRONTEND**. User scope amendment
SHA256 `cc874517b5366c019b625ce3d6fcf0a33333c359cfb1dd8d03039d2cde290e99`.
Historical M1 remains BLOCKED_FRONTEND_QUALITY; the new Train-only representation
and predicted-memory policy branch is authorized independently. Native MASA is
LIMITED_COVERAGE_PROVENANCE_INCOMPLETE_FRONTEND, annotated-cadence replay only.
Old R0/R1 artifacts and negative conclusions remain unchanged. This is NOT R2.

## Actual data and role partition

Current Train SHA256 `7eb551fdeeeebc76b876ae255f91dc5662c7270a125955c5f1be2d9bd30921d0`:
500 videos,18274 annotated images,2196 Known tracks/48 categories/43380
observations.1170 tracks have>=16 observations; the1026 shorter tracks remain
eligible. No Val/Test data or descriptor/metric-based split selection.

| Purpose | Tracks | First-at-most16 observations | Role |
|---|---:|---:|---|
| Representation fitting | 1305 | 14714 | 15 Train Known classes |
| Prototype | 151 | 1907 | 48 legal Train Known classes |
| Development | 258 | 2949 | Fit Known + 4 disjoint pseudo classes |
| Policy meta-training | 229 | 2586 | Fit Known + 4 other pseudo classes |
| Final heldout selection | 223 | 2472 | Fit Known + 4 other unseen pseudo classes |

All five partitions are globally video-disjoint. The representation,development
pseudo,policy pseudo and final pseudo category sets are disjoint. Known probes
intentionally share fitting classes, not purported category holdout. Preserve
short first chronological prefixes using min(cap,available); report actual count
and sparse category support.30 remaining tracks cannot enter a lawful selected
role after video/category reservations and are recorded, not silently dropped.
48/78 final inherited Known IDs have Train prototypes;30 missing IDs never get
Val-GT replacements and remain in final fixed GT denominators. Episode Known
prototypes include only fitting classes, never the episode's pseudo classes.

## Frozen common input and controlled first training matrix

Reuse frozen DINOv2 ViT-B/14,768D,existing exact518bilinear/context.1 crop and
normalization,TF32off,FP16 compact storage. Existing compatible smoke/pilot
observations are reused by source/protocol/payload/image+box+frame identity;
old per-image raw-byte hashes are unavailable and are not retroactively claimed.
New encoding is image-grouped and hashes current input bytes; no crop files,
million JSONs, model optimizer, detector/tracker rerun or semantic filtering.

T1: three seeds1027/1028/1029,1000 AdamW steps per model,LR3e-4,decay1e-4,
clip1,temperature.1,8categories×2different-video tracks per batch. Train only
the1305 fit tracks. Every anchor has a cross-video positive and negative.
Group variable actual prefix lengths rather than padding or exposing future
descriptors. Same batches,adapter initialization,optimizer and budget per seed.
No synthetic corruption or extra reliability loss in the main A1/A2 comparison.

Plain A1 and two finitely registered teacher Gram weights1/5 establish a
geometry-preservation control on the enlarged independent dataset, not a
renamed rerun of R1. Checkpoints250/500/1000; select on development only using
mean macro cross-video Rank1 and Known/pseudo prototype-score AUROC. Tie-break
earlier step/lower geometry weight. Raw A0 has no optimizer. No hidden trials,
final-heldout-driven architecture change or Val Novel selection.
Geometry family selection uses the mean development score across all three
seeds and a lower-weight tie-break; all seeds share the selected weight. Every
trial and checkpoint remains in the report, not a per-seed favorable method.

T2: existing causal CategoryEvidence and a static-mean capacity-matched residual
control (same16833 extra parameters). Match all A1 observations,pairs,init,
optimizer and budget. Additional reliability supervision is a separate ablation,
not an architecture claim. A2 may be dropped if no independent stable gain.
For the paired architecture comparison, use the selected A1 checkpoint step
for A2 and the capacity control, not a later favorable checkpoint. Initial
adapter and entire actual batch/prefix digests must exactly match per seed.

T3 policy is distinct from previous GT representation exception and is now
explicitly authorized. D1/D2 share candidates/capacity, use predicted-memory
rollouts and cost-bearing WAIT. Labels supervise loss only; memory is never
GT repaired. Detailed policy source/loss constants must be registered before
first policy fitting or development results. Compare at comparable coverage.

## Delivery and limitations

Each stage: tests→commit→push→verify exact remote→next stage. Features,
checkpoints,raw GT and NPZ stay private. New storage soft15GiB/hard30GiB;
retain>=25% system RAM,4GiB free disk/GPU reserve; no foreign interference.
Freshly check UUID/free VRAM rather than reuse historical GPU numbers.

T4 uses audited standard/persistent evaluators,all four orders and five prefixes;
Hungarian only for Standard OCD. T5 begins after candidate freeze, uses existing
MASA988/36375 annotated-frame stream without physical reruns, all unmatched
tracks, fixed4413/819 GT denominators and527/529/547/531 reuse opportunities.
Both stronger frontend qualification and exact MASA all-stage provenance remain
unmet. T0 engineering/metadata PASS is not a trained model or scientific PASS.
