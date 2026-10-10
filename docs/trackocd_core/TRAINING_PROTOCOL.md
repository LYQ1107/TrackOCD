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

## T1 actual training and pre-heldout policy registration

Actual T1:9fits,3seeds x3weights,1000steps each,276.5689s wall,56,775,888B
checkpoints retained locally. All shared initial-adapter and complete batch
digests match. Development-only mean scores plain0.606502/Gram1 0.596447/
Gram5 0.585495 select plain A1 globally. Selected steps500/250/500 for seeds
1027/1028/1029. No final heldout or Val/Test used. GPU peak reserved96,468,992B,
RSS1,192,591,360B. Training completion is not scientific improvement.

Policy source is now registered before policy fits or final-heldout results.
D1/D2 share the17-32-4 MLP708parameters,candidate feature definitions,initial
weights,descriptor order and budget. Candidate geometry/state may diverge
after their own predicted errors; it is never aligned by GT. Inputs are top2
Known/anonymous similarities,margins,uncertainty,maturity,quality,elapsed,
predicted-memory dispersion,count,reliability and candidate-presence masks.
Known chooses an actual legal ID,Existing chooses the actual top existing
token,New monotonically allocates S:id,Wait leaves memory untouched.

Train only229policy tracks,20epochs x4orders,cycling5caps,80AdamW updates per
fit. Frozen raw/A1-selected/A2 families x2policies x3seeds; no representation
updates. D1 nominal action CE; D2 adds expected costs and delayed credit to
prior predicted writes when later candidates encounter actual absorbing
mixed-category history. External GT bookkeeping supervises losses only.
All costs,capacity,budget,checkpoint5/10/20 and WAIT bias[-2,-1,0,1,2] fixed
in persistent_policy_training.json. Development chooses checkpoint and
coverage-target points; report realized coverage,no false matched-coverage
claim unless BOTH fixed pseudo-Novel reuse-opportunity commitment coverage
and total target commitment coverage differ by<=0.05. Operating points target
the former, not Known-dominated total coverage. WAIT is censored/unresolved at each independent
capped-prefix replay, with positive observation-dependent cost,not a proven
dense frame-online delay controller. Short tracks use their actual count.

GT evaluation source/config is fixed before final-heldout results:all finite
representation trials,3training seeds,4orders,5caps;207Known/16pseudo-Novel
targets. Thresholds use the same25point development/main/p16 grid per method
and representation; PHE remains INCOMPARABLE. Retrieval Recall@1/5/10 and
spectral rank complement ACC,AUROC,CT,all-Novel wrongKnown and pollution.
Mean/std across training seed means are distinguished from order variation.
No heldout-driven redesign or Val Novel tuning is authorized.

## T3 actual frozen policy fits

All18registered raw/A1/A2 xD1/D2 x3seed fits completed:20epochs x4orders,
80updates each,54private checkpoints totaling313488B,446.9464s oneCPU,
peak847216640B RSS. Paired initialization and descriptor-order/prefix hashes
match. Representation weights stayed frozen and actual predicted memory was
never GT repaired. Most zero-bias development H/CT scores are0; bounded WAIT
biases can recover commitment coverage but cannot be called correct discovery
without the frozen heldout/fragmentation/merge evaluation. No extra trials.
