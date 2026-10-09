# MASA and AED physical frontend preflight

2026-10-10, Asia/Shanghai. MASA-SAM-B is a conditional low-cost image-only
candidate, not a qualified or recovered primary frontend. Its default TAO
configuration cannot be used unchanged. AED's default release also lacks the
required inherited Known-only supervision and detector provenance binding.
M1 remains `BLOCKED_FRONTEND_QUALITY`; no frozen-primary marker is written.

## MASA proposal and supervision route

The [official training instructions](https://github.com/siyuanliii/masa/blob/c5472b9c7615f35abdf1188cb1a0c5408fe50d66/docs/train.md)
describe 500K raw SA-1B images and SAM-generated regions. The
[model zoo](https://github.com/siyuanliii/masa/blob/c5472b9c7615f35abdf1188cb1a0c5408fe50d66/docs/model_zoo.md)
reports no in-domain images during MASA training, but its TAO association
numbers use public Detic-SwinB detections. Those numbers are not evidence of
anonymous native proposal coverage or our HOTA/Commit-CT result.

The [paper's inference route](https://arxiv.org/html/2406.04221v1)
retains the object-distillation detection heads to generate proposals without
semantic classes. This motivates inspecting the released SAM-B one-class
RPN/ROI heads, not fetching Detic detections or class text. Complete release
tensor coverage and runtime exclusion still need verification.

The pinned [TAO Val configuration](https://github.com/siyuanliii/masa/blob/c5472b9c7615f35abdf1188cb1a0c5408fe50d66/configs/masa-sam/open_vocabulary_mot_test/masa_sam_vitb_open_vocabulary_test.py)
has `load_public_dets=True`, an external Detic path, one-class RPN/ROI heads,
and category-agnostic association. Setting only that flag to false does not
establish a native proposal runner: the no-public/no-given branch in
[MASA.predict](https://github.com/siyuanliii/masa/blob/c5472b9c7615f35abdf1188cb1a0c5408fe50d66/masa/models/mot/masa.py)
dispatches `self.detector.with_neck` and `self.detector.predict`, whereas
[SamMasa](https://github.com/siyuanliii/masa/blob/c5472b9c7615f35abdf1188cb1a0c5408fe50d66/masa/models/detectors/sam_masa.py)
inherits `BaseModule` and supplies neither detector API. The proposal heads
belong to MASA, not that SAM detector. A small explicit multi-level-feature
RPN/ROI wrapper would be a **new candidate implementation**, not recovery of
old SimOWT or reproduction of the public-Detic protocol.

## Causal output requirement

The [demo postprocessor](https://github.com/siyuanliii/masa/blob/c5472b9c7615f35abdf1188cb1a0c5408fe50d66/demo/utils.py)
uses centered box smoothing, full-segment mean scores and whole-track removal
based on any later giant box. All three can change earlier outputs using
future frames. The [demo](https://github.com/siyuanliii/masa/blob/c5472b9c7615f35abdf1188cb1a0c5408fe50d66/demo/video_demo_with_text.py)
enables postprocessing unless `--no-post` is supplied. A causal candidate must
bypass the entire postprocessor, not merely discard categories afterward.

A synthetic source-formula witness changes only coordinate index 2 from
0 to 10 in a five-frame segment: centered width-5 smoothing changes the
first coordinate from 0 to 2. This demonstrates a future dependency in the
inspected formula, not execution of the upstream model, an actual TAO stream,
or proof that historical MASA results used this demo path.

## Dependencies and conditional asset budget

The [documented installation](https://github.com/siyuanliii/masa/blob/c5472b9c7615f35abdf1188cb1a0c5408fe50d66/docs/install.md)
uses MMDetection 3.3.0 and MMCV 2.1.0. Its
[environment](https://github.com/siyuanliii/masa/blob/c5472b9c7615f35abdf1188cb1a0c5408fe50d66/environment.yml)
uses Python 3.11, torch 2.1.2/cu118 and NumPy 1.26.4. Existing A100 torch
2.6/cu118 and NumPy 2.2 are not a verified replacement; mmengine/mmcv/mmdet
are absent. The requested official cu118/torch2.6 wheel index returns 404,
while the torch2.1 index lists a cp310 MMCV 2.1.0 binary wheel. The installed
compiler is CUDA 13.2. No source compilation, toolkit installation, base-env
downgrade or wholesale requirements installation is planned.

At pinned [official weight repository revision](https://huggingface.co/dereksiyuanli/masa/tree/25ed372c47f2c46cf36fd446d1b657b656bc7ea9),
LFS metadata lists `sam_vitb_masa.pth` as 558,882,875 B and converted
`sam_vit_b_01ec64_mmdet.pth` as 375,042,767 B: a conservative 933,925,642-B
weight ceiling at initial preflight. The subsequent first-checkpoint recovery
below verifies local bytes and defers the second until exact model key matching.
Exact filenames, LFS SHA256 and revision are in `masa_sam_candidate_smoke.json`.

The conditional smoke uses a separate runtime only after an exact minimal
binary-wheel resolver lock and size check. Python 3.10 is a candidate supported
by the observed wheel, not a verified upstream-equivalent environment. All
new environment/source/weights/output must fit **8 GiB** and the goal's total
15/30-GiB limits; 8 GiB is a stop ceiling, not a measured installation size.
Fresh disk evidence is 93,845,364,736 B available and RAM 108,202,061,824 B
available. Existing named new payload remains 531,287,428 B plus small files.

After synthetic import/shape checks, at most eight already-local Train images
from one video may enter a 600-second single-worker smoke. No GT boxes,
categories, track IDs, public detections, text, future frames or offline
postprocessing enter the runner. Keep the release's proposal thresholds;
no training, threshold search, new representation correction or full Val/cache.
Unsafe pickle fallback is forbidden. Empty/failing results must be retained.

## AED default release boundary

The [default TAO config](https://github.com/balabooooo/AED/blob/e9c0c7f1884fdcf76c24747d4f3e8245dcfb1064/configs/tao.args)
sets `train_base=True`. Its [dataset code](https://github.com/balabooooo/AED/blob/e9c0c7f1884fdcf76c24747d4f3e8245dcfb1064/datasets/tao_dataset.py)
filters LVIS rare categories, not an explicit inherited 78-Known whitelist,
and consumes GT track identity during training. The actual role overlap and
release training binding have not been checked, so this is **not proof of
forbidden supervision**. It is insufficient for drop-in qualification.
Default public-detector vocabulary/supervision need their own audit. No AED
weights or annotation pack are downloaded; an association-only replacement
on unchanged PANDAS detections would not fix their missing-target coverage.

## Verification and next action

Fifteen selected files, **96,708 B**, match pinned Git blobs and SHA256. The
source-only audit took 0.0066 s, 17,864-KiB peak RSS, one CPU worker. All
**137 core and recovered-v2 tests pass**; this is engineering evidence only.
Private upstream copies are excluded from Git; public aggregate receipt,
source manifest, tests and conditional smoke configuration are included.

Initial next action was binary-wheel resolution and safe checkpoint inspection;
both completed in the subsequent recovery below. Isolated installation,
exact model key matching and the bounded proposal smoke remain unverified.
Smoke success would still require separate physical/coverage evaluation
before M1 qualification. Preserve Q0, all old streams/shards and the negative
GT representation findings. The one correction round remains consumed;
GT policy-learning scope is still awaiting explicit clarification.

## Frozen checkpoint recovery and exact wheel budget

The first checkpoint is now local: **558,882,875 B**, SHA256
`441c05bf9519632fead1afd5200bb6a4f13b4a41c58f4428024490b4c2bd777c`,
matching the pinned repository LFS identity. Download took approximately
117 seconds; the verified partial file was renamed without overwriting an
existing destination. No second SAM pretrain weight was downloaded.

Read-only `torch.load(weights_only=True, mmap=True, map_location='cpu')`
loaded **419 tensors** without extra global allowlisting or unsafe fallback.
Components include 177 backbone tensors, 75 adapter tensors, six RPN tensors,
eight ROI tensors and 16 track-head tensors; prompt/mask components account
for the remainder. The ROI classification weight is `[2,1024]` (one foreground
plus background), regression `[4,1024]`, and SAM blocks 0–11 are present.
This is structural evidence, **not exact runtime model compatibility, numeric
finiteness, supervision certification or M1 qualification**. Checkpoint
metadata/category names never became model input or public output. Inspection
took 3.23 s with 355,360-KiB peak RSS; no model was constructed or executed.
The subsequent safe metadata check finds no saved training configuration
and zero metadata keys. Official generic-image/no-in-domain documentation
supports candidate investigation, but the checkpoint does not itself provide
a complete released-stage training ledger. Do not infer that ledger from
one-class tensor shapes or silently invoke COCO metadata defaults.

The resolver generated an exact 48-package binary-wheel lock for Python 3.10
and the documented torch2.1/cu118 family. No packages were installed. The
official cu118 index identifies the working `download-r2.pytorch.org` URLs;
the initial connection failures do not mean cu118 is unavailable. The wheel
inventory uses bounded ZIP-directory ranges and curl for the three PyTorch
wheels whose Python HTTP requests returned 403. Small archives may be read
fully in memory within 64 KiB; no archive payload is saved or installed.

All **48 selected wheel directories** were measured: **5,378,584,576 B**
of unpacked 4-KiB allocation, plus **1,344,152,372 B** reserved for environment
overhead/bytecode. Including both optional weights and a 16-MiB source/output
reserve gives **7,673,439,806 B**, below the 8-GiB candidate ceiling. Network
directory bodies total **4,407,901 B**; inventory took 61.11 s with
28,280-KiB peak RSS. Archive-content SHA verification is still required at
installation; range metadata is not cryptographic payload verification.

This is a **resident budget estimate**, not a verified installer transient
peak. Use a same-filesystem temporary area, no retained wheel cache, and a
monitored single-package/hardlink installation so unpack/copy/download peaks
cannot silently exceed the ceiling. Do not duplicate an existing destination,
compile sources or install unpinned/text/benchmark extras to make imports pass.
The upstream package initializer imports dataset and semantic modules;
the native runner must explicitly register only the inspected required image
modules, without calling the default loader's class-name metadata fallback.

Public recovery receipt, binary lock, directory budget and synthetic tests
are retained. Actual new named weight/dependency/GT/checkpoint payload is now
**1,090,170,303 B**, excluding small source/report files and unchanged existing
assets. No inference/training, GT/Test access, base-env changes, foreign-worker
interference, further representation correction or primary freeze occurred.

Next: guarded isolated binary installation, source-whitelisted native model
construction and strict component key/shape matching. Only then execute the
already limited eight-Train-image causal smoke, preserving failures/empties.

## Actual isolated runtime and bounded native smoke

2026-10-09 21:30 UTC / 2026-10-10 Asia/Shanghai. The isolated runtime and
native engineering smoke are now **PASS**; M1 remains
`BLOCKED_FRONTEND_QUALITY`, with no primary freeze or formal physical metrics.
The original conditional plan above remains immutable historical preregistration;
actual results are in `masa_runtime_install_summary.json` and
`masa_native_smoke.json`.

All **48 hash-locked binary wheels** were installed into
`/data3/liuyeqiang/.venvs/trackocd-masa-smoke`, one package at a time, using
same-filesystem hardlinks and no index/dependency resolution, source build,
toolkit installation or base-env change. Installed RECORD SHA256 checks cover
**20,423 files / 5,325,373,594 logical bytes**, with zero mismatches. The base
distribution snapshot is unchanged. Peak owned allocated size during install
was **5,945,593,856 B**, measured rather than inferred from ZIP metadata.

The first Torch install succeeded, but uv's `direct_url.json` had empty
`archive_info`; the initial checker incorrectly required that optional saved
hash. The original partial environment/log were preserved. Resumption checked
the exact requirement, successful hash-required operation/control flow, source
URL/version and all installed RECORD digests, without downloading Torch again.
Later operation receipts are saved before metadata inspection. RECORD integrity
is not independent archive re-hashing; an empty saved metadata hash is never
trusted without the verified owned hash-required operation. Both attempts are
retained; their install wall times were 462.48 and 373.54 seconds.

Fourteen additional image-only source/config/license files, **125,292 B**, match
the pinned upstream Git blobs. Explicit module registration avoids MASA's
dataset/semantic package initializers. The wrapper does not call `init_masa`,
`MASA.predict`, public detections, checkpoint class-name defaults, dataset
loaders, the offline smoother or the track-wide filter. It uses current RGB
pixels/geometry and current ordinal, SAM multi-level features, the released
adapter/RPN/one-class ROI heads, and the unchanged past-memory tracker.
Network and implicit external-child execution are denied inside the worker.

All **419 model keys and shapes match exactly**, all state tensors are finite,
and every parameter is frozen. No second SAM checkpoint is needed. CPU binary
NMS/RoIAlign, synthetic preprocessing and empty-proposal fixtures pass. An
initial model-load attempt failed before any real-image forward because torch
2.1's mmap interface requires a string filename rather than `Path`; the owned
wrapper now passes `str(checkpoint)` without unsafe pickle fallback. That
12.28-second / 1,152,876,544-B-RSS failure is preserved in the public receipt.

The completed smoke used **four unique existing Train images from one video,
two sequential replays, eight forwards total**. Image selection reused the
previous long-Known-GT pilot plan and is biased; no GT/semantic fields or image
paths become model input. Frame ordinals are subsampled-video units, not original
frame indices. The second replay inverts only the last two images. Both first
frames' detection/track arrays are **exactly identical** across replays, and
the future pixel hashes differ. This is a bounded causal witness, not a universal
causality certificate or temporal scientific PASS.

Every frame has **50 ROI detections**, hitting the inherited cap without tuning.
Original tracking counts are **43/37/40/42**; changed-future counts are
**43/37/40/40**. All outputs are finite/nondegenerate, native foreground label
0, exported as anonymous category 1. No boxes were repaired or postfiltered.
Counts do not establish recall, useful coverage, purity or tracking accuracy;
the proposal-cap saturation must remain visible in the next evaluation.
The upstream tracker parameter `memo_tracklet_frames=10` is preserved, but its
empty-frame early return and pruning-after-association are not silently repaired
or described as a guaranteed pre-association age bound.

Worker time was **15.31 s**, supervised wall time **16.88 s**. Peak host RSS:
**1,687,347,200 B (1.57 GiB)**. Peak GPU allocated/reserved:
**3,126,221,312 / 3,439,329,280 B**. Fresh GPU selection preceded the owned worker;
sampled total selected-device memory reached 3,815 MiB. Candidate allocated size
including retained setup/smoke attempts is **5,945,954,304 B (5.54 GiB)**, below
8 GiB. Prior other named payload is 531,287,428 logical bytes; the mixed
known subtotal 6,477,241,732 B is not an exhaustive filesystem inventory.

Regression: **177 core/recovered-v2 tests pass**. Only small code/config/aggregate
receipts are published, not wheels, weights, raw images, GT rows or source copies.
No training, Val/Test job, threshold search, R2, GT policy expansion or formal
M1/M9/M11 PASS occurred. Complete checkpoint-stage supervision provenance and
an explicit bounded physical coverage/pollution evaluation remain required.
Do not automatically rerun the completed smoke or escalate to full Val/cache.

## Published supervision trace and fixed bounded physical diagnostic

Continuation 2026-10-10: six pinned source/doc files (**32,848 B**, four new
files **25,116 B**) seal the published training-route audit. The training data
template has one SA-1B leaf, generic bbox annotations and synthetic paired-view
augmentation. Both converter bbox/segmentation rows assign category 1; dataset
metadata is generic `object`. The model zoo explicitly links `sam_vitb_masa.pth`
and describes no in-domain training. These are published route assertions,
not a checkpoint-saved training ledger. The complete pinned tree has **no
SAM-B training configuration**; the documented `convert_sa1b_to_coco.py`
filename is absent, while `convert_sam_2_cocofmt.py` is present. The supplied
train model is GroundingDINO, not an exact SAM-B release configuration.
`masa_training_route.json` preserves this missing binding without claiming
forbidden supervision. No dataset converter, training script or data pack ran.

A separate M1 diagnostic is preregistered in `masa_physical_diagnostic.json`:
ascending Val video IDs **4/20/22/23**, each first 16 chronological image-metadata
rows, **64 images maximum**. Selection never inspects GT category/track/annotation
fields, substitutes a video, or expands based on later Novel support. Current
Val GT is byte-identical to the prior PANDAS canonical GT symlink: SHA256
`0414885ee2702c2d3176cf6184e7811a7bd1c1347a157fef57a91020976776ee`.

This is **not a rerun of the consumed eight-forward smoke**, full-Val/M9,
semantic model selection/training, tracker tuning, a second representation
correction or primary freeze. Released weights and native model config are
unchanged. A remote-verified preregistration is required before execution.
The prediction worker denies GT/role/Test reads and network/external children;
only current pixels/geometry/ordinal reach the frozen model. Four compact NPZ
files are sealed before a fresh evaluator opens GT. Evaluation uses the existing
unchanged canonical TAO_OW adapter (all classes, cap300), fixed temporal-IoU
coverage matching, annotated-clip lengths and posthoc category/identity mixing.
Unknown unmatched observations cannot certify purity or prove background.

The frozen PANDAS FG-0 projection uses identical scoring images and GT, but
retains its dense-frame association/prehistory. Native association starts at the
first selected annotated image and uses annotated-only ordinal cadence. This
is a physical-route diagnostic, **not a matched-input association ablation**.
Length statistics refer only to the selected projection, not whole-video track
lifetimes. No four-clip result, however favorable, can establish full-Val coverage,
universal causality, legal released-stage supervision or M1 qualification.
Zero role denominators remain null, and all empty/failing clips are retained.
No TETA/OWTA is invented. An actual unchanged-canonical-adapter synthetic
perfect-track fixture has HOTA/AssA/DetA/DetRe=1. Preregistration regression:
**185 tests pass**.

## Actual fixed Val4 physical diagnostic results

The **same remote-verified preregistration `1d0f3a1`** executed once: video
IDs4/20/22/23, 16 images each, 64 frozen forwards, no resampling/tuning.
Prediction was sealed before independent GT evaluation. Both streams used
the same 328 GT annotation rows and unchanged pinned canonical TAO_OW/HOTA
code. This is a **four-clip diagnostic**, not full-Val or a matched-input
association/semantic contribution comparison.

| Frozen physical route | Clip HOTA | AssA | DetA | DetRe | Known clip coverage | Novel clip coverage |
|---|---:|---:|---:|---:|---:|---:|
| MASA native candidate | 0.169655 | 0.502096 | 0.059033 | 0.531611 | 8/26 (30.77%) | 0/2 |
| Existing PANDAS BT-FG-0 projection | 0.122697 | 0.372579 | 0.041192 | 0.583440 | 10/26 (38.46%) | 0/2 |

These combined HOTA values come from canonical count-weighted sequence
combination, **not averaging four independent HOTA numbers**. Numeric scores
are fractions, not percentages. The receipt retains each video's real metrics,
all alpha arrays and denominators. PANDAS whole-Val HOTA remains its separate
historical 0.110113 result; neither clip value overwrites it.

All MASA ROI frames saturate the unchanged **50-proposal cap**. There are
495 annotated-clip physical identities / 2,843 observations, mean length
5.7434, median4, p90=14 and **119/495=24.04%** single-observation identities.
The PANDAS projection has 2,807 identities / 4,543 observations, median1,
**2,013/2,807=71.71%** single-observation identities. Different association
cadence/prehistory remains material: this cannot isolate a tracker-architecture
effect, and clip lengths cannot be called whole-video lifetimes.

At the fixed frame IoU threshold, MASA matches **184/2,843** observations to
GT, leaving **2,659 unknown**; PANDAS matches **216/4,543**, leaving **4,327
unknown**. Neither has observed multicategory mixing among these matches, but
that does **not** establish pure track representations: MASA has 456 identities
without any GT match and 477 identities with unknown observations. Only18
MASA observed identities have every observation matched to one category.
Six MASA identities touch multiple GT individuals; this is distinguished from
category mixing, not interpreted as different-class contamination automatically.
Unmatched observations remain unknown under incomplete annotation, not proven
background or reliable foreground. Canonical preprocessing removed zero selected
rows for both routes; no private unmatched-filter improves the reported scores.

Interpretation: native candidate association/fragmentation diagnostics are less
poor in these clips, but Known reliable coverage is lower and both Novel clip
targets remain missing/unreliable. **No primary qualification PASS** is supported.
The two-Novel-target sample cannot establish a full-Val Novel recall FAIL either.
It is not permissible to select more favorable clips, raise the proposal cap,
retune thresholds or infer category-discovery contribution from this comparison.
M1 remains `BLOCKED_FRONTEND_QUALITY`; full qualification and released-supervision
scope still require stronger evidence. No frozen-primary marker is written.

Frozen worker **27.74 s**, supervised **29.74 s**; peak host RSS
**1,669,230,592 B**, GPU reserved **3,441,426,432 B**. Independent CPU evaluator
**5.60 s**, **847,216,640-B** peak RSS. Four private compact NPZ files total
**118,457 B**; all bounded diagnostic artifacts occupy **2,723,840 allocated B**.
The owned candidate including retained install/smoke/diagnostic directories
is **5,948,731,392 allocated B**, below8GiB. No other processes remain ours on
GPU, no training/Test/semantic input/extra weight or full-Val cache ran.
Public evidence: `masa_physical_diagnostic_result.json` and
`masa_physical_prediction_summary.json`; GT subsets and raw NPZ boxes stay
private. Result regression: **187 tests pass**. The completed run is preserved;
do not silently rerun it or broaden its result scope.

## Separate all-universe M1 physical audit preregistration

The original expanded FINAL_GOAL M1 explicitly requires Val physical quality
and target-coverage auditing. After the completed Train8 engineering and Val64
fixed diagnostic, a **separate physical-only all-Val preregistration** is now
prepared. This is not an expansion/rerun of the old smoke/clip protocol, M9 OCD
evaluation, semantic model selection/training or full DINO feature caching.
The M8 small-Train-first scientific gate remains unchanged. No new authority
is inferred from smoke success itself: scope comes from the expanded M1 goal.

Image-metadata-only inventory confirms **988 videos / 36,375 annotated frames**,
all present locally, strict chronological/nonduplicated image IDs, no path
escape or Test/Train substitution. Existing image bytes **4,204,120,627** are
reused in place; no image/model transfer. Every required image is hash-sealed
in the private plan (SHA256
`bb4ae9b1ce346d490021ab2b70a5e35078e38134184a2bed58d66d5af12ac179`).
Preparing all hashes took107.45s, peak RSS847,216,640 B. The public config
contains counts/hashes/protocol, not raw image paths or semantic names.

Same MASA weight/native config/score.02/cap50, no rescue tuning or physical
training. At most4 freshly idle GPUs, each8GiB GPU/4GiB host ceilings, system
RAM headroom25%, disk headroom4GiB. Uncompressed prediction payload bound is
88,464,000 B; prediction/evaluator artifacts conservative ceiling1GiB,
candidate ceiling8GiB, overall soft15GiB/hard30GiB. Before launch, measured
candidate5,959,065,600 allocated B and conservatively the **entire repository
plus isolated environment**6,438,285,312 B (including pre-existing code, so
not presented as an exact incremental storage ledger). No arbitrary hourly
completion cutoff; genuine resource/input/immutable-state failures stop only
newly created owned workers and preserve evidence.

Predictions are video-atomic: SHA/config/full-video input-plan/array checks
must pass before any existing complete video is reused; empty outputs remain
valid. Incomplete attempts are retained; a restarted incomplete video's
causal state must replay from its first image and real attempts are counted.
Existing four short clips do not contain full-video tracker state and cannot
masquerade as complete video shards. At each new complete-video boundary, all
frozen tensor values are verified unchanged. Independent GT scoring is allowed
only after all988 complete markers and36,375 frames are verified and sealed.

One-CPU evaluation uses the same pinned canonical TAO_OW/HOTA, entire original
GT universe/role denominators, unchanged partial-annotation preprocessing and
count-weighted sequence combination. A nontrivial synthetic two-video fixture,
including empty predictions, verifies per-video canonical evaluation+combination
equals a single full-universe adapter call. Only current-video JSON scratch
is kept, avoiding multi-million-row duplicated GT/pred JSON. Posthoc purity
keeps unknown rows and distinguishes category from individual mixing.
PANDAS projected scoring shares GT/images, but its association input cadence
and dense-frame history differ: not an isolated/fair tracker-architecture ablation.

Public supervision evidence is graded at **official publication/pinned release
level**, not claimed as a private cryptographic training ledger. The
[MASA paper sections4.1/J.1/J.2](https://arxiv.org/html/2406.04221v1) describe
main SAM models' generic SA-1B image route, frozen SAM foundation and a final
track-head-only training phase. Separate ablation/domain-adaptation models
cannot automatically establish this release's lineage. The
[SAM contributor statement](https://github.com/facebookresearch/segment-anything/issues/53)
supports SA-1B foundation pretraining. The exact SAM release-stage config/data
binding remains absent and disclosed, not evidence of forbidden supervision.
No automatic primary freeze or M9/semantic PASS, regardless of this audit's scores.
All preregistration tests now pass: **207 passed, no skips**. Commit/push/exact
remote-HEAD verification is still required before actual GPU inference.

## Actual sealed full-Val physical inference, evaluation still pending

Remote-preregistered `455e9dda98752323a1cb12c1e3b2c0fcf9e31900` was independently
verified at **2026-10-09T22:16:15Z**, before the new owned run. Four freshly idle
GPUs completed the exact988-video/36,375-image universe once; all4 workers
returned0. Supervisor2637.60s (~44min), no wall-time cutoff, foreign-process
interference, training, Test, parameter change or new image/model transfer.

Sealed stream: **1,540,022 track observations / 1,811,677 raw detection rows**,
988 complete video markers, private compressed NPZ total **63,222,355 B**.
The original cap50 is saturated on **35,934/36,375 frames (98.79%)**; zero
empty detection/tracking frames. These counts are not accuracy or coverage.
All419 frozen tensors match the release and remain value-identical at every
complete video and final worker boundary; all4 initial state digests also match.
All registered arrays/metadata/hashes were reverified before full stream seal.

Peak per-worker RSS1,689,985,024 /1,689,075,712 /1,689,571,328 /1,691,398,144 B;
GPU reserved maxima3,445,620,736 /3,447,717,888 /3,445,620,736 /3,447,717,888 B.
Supervisor candidate allocation peak **6,032,039,936 B**, below8GiB; observed
RAM headroom remained ~85%. Generated predictions and state/attempt evidence
stay private. Small public `masa_full_val_prediction_summary.json` carries
counts/hashes/frozen-model/resource proof, not raw boxes, labels or image paths.

This phase is **successful frozen inference, not M1 qualification or M9**.
At this entry, independent one-CPU canonical full-GT evaluation is started
only after the sealed-prediction checks; no full-Val HOTA/coverage/purity result
is yet claimed. No downstream physical/semantic tuning or primary freeze follows
from output counts. Old8/64 results remain intact and separately scoped.
Sealed-prediction receipt regression: **208 tests pass, no skips**. Independent
CPU evaluation is still running at this inference-stage delivery.

## Actual full-Val physical audit result (988 videos, not M9 semantics)

Independent evaluation completed after the full prediction seal: **988 videos /
36,375 images / 113,112 GT annotation rows**, no dropped videos or narrowed role
denominators. Same pinned canonical TAO_OW/HOTA and count-weighted sequence
combination. One CPU process, no GPU or model training: **1,212.44s**, reported
peak RSS **847,216,640 B**. PANDAS projection reproduces frozen exported
**HOTA exactly, absolute difference0**; all four physical metric values also
match the previously recorded reference. No historical reference is overwritten.

| Physical route | HOTA | AssA | DetA | DetRe | Known reliable coverage | Novel reliable coverage |
|---|---:|---:|---:|---:|---:|---:|
| Frozen MASA native candidate | 0.145054 | 0.440879 | 0.048489 | 0.631809 | 1,400/4,413 (31.72%) | 189/819 (23.08%) |
| Frozen PANDAS BT-FG-0 projection | 0.110113 | 0.330246 | 0.037554 | 0.578390 | 1,442/4,413 (32.68%) | 30/819 (3.66%) |

Novel reliable target count is **6.3x** the PANDAS count, but **630/819 Novel
targets remain missing/unreliable**. Known reliable targets are42 fewer than
PANDAS. These are physical-route improvements, not evidence of semantic
discovery, an independent temporal-evidence effect or a matched-input tracker
ablation: input cadence/detection source/history differ and remain disclosed.
Do not extrapolate the old four-clip0/2 result to this full-Val189/819 result.

Native annotated-cadence stream: **304,561 physical identities**, mean length
5.0565, median2, p90=13, p95=22; **114,380/304,561=37.56%** single-observation
tracks, **71.55%** at most4 and **92.52%** at most16 observations. PANDAS
annotated projection: **1,041,719 identities**, mean1.7580, median1, p90=3;
**728,116/1,041,719=69.90%** singles. These projection counts must not overwrite
PANDAS dense-lifetime2,370,335-track/10.19%-singleton history. Native fragments
are materially fewer, but still substantial; short tracks cannot disappear from
later prefix evaluation or headline denominators.

Native geometry-matched rows **83,730/1,540,022**, leaving **1,456,292 unknown**.
There are **290,240** identities without any GT match and **298,926** with
unknown observations; **391** observed multicategory identities, **1,761**
touch multiple GT individuals, and only **5,514** have all observations matched
to one category. The matched-only majority fraction0.98980 is **not** a purity
certificate. PANDAS has77,254 matched/1,754,137 unknown rows,626 observed
multicategory/1,727 multi-individual identities and28,422 entirely observed
single-category identities. Unmatched objects remain unknown under incomplete
annotation, not proven background/false positives. No unknown-row filter repairs
either candidate's quality before scoring.

Canonical preprocessing evaluates **1,444,087 native** and **1,703,763 PANDAS**
rows, removing95,935 and127,628 respectively under its original rules. Raw
coverage/length/purity retain all rows; this removal is not a private unmatched
filter or an input to the semantic model. Entire NPZ hashes/bytes and each
video's score/coverage/purity are retained in the aggregate result receipt.

**Primary freeze remains unresolved**, not automatic PASS or a newly invented
numeric failure threshold. The original goal asks for a reliable frontend but
does not specify a numerical coverage gate. This is the best currently measured
legal-image-route candidate, yet coverage/fragmentation are limited and the
exact private release training ledger remains undisclosed. User direction was
requested whether to accept an explicitly limited-coverage main benchmark or
keep the reliable-frontend requirement and leave primary unfrozen. No threshold,
cap, sampling, weight, R2, GT-policy scope, DINO cache or training is changed
while this decision is pending. The original/R1 negative semantic feasibility
results stand; no M9/main scientific PASS or final-goal completion is claimed.

Final regression: **209 passed, no skips**. After both real jobs terminate,
private full prediction+evaluation allocation **72,204,288 B**; candidate named
roots **6,034,083,840 B** and conservative entire repo+isolated env
**6,517,043,200 B**, below the stated ceilings. Public aggregate result is
**3,367,976 B**, SHA256
`5d4e2d903204bf6313f6df42a300171d3596413f34ebaf2d731ed4fbb466b220`.
No raw NPZ, GT payload or model weight is staged. No owned local job remains live.

## Preregistered tiny real predicted-box visual-interface check

The remaining M1 visual-interface gate is checked separately, not by promoting
the completed Train Known GT cache. The first metadata video and its first-image
four numerically sorted physical IDs supply at most two observations each,
scanning at most eight images. No label, reliable-coverage or future-length
selection; empty/short tracks are never replaced. Existing sealed physical
shards are read, not re-inferred. Maximum **8 real crops / one fresh idle GPU /
batch4 / 4GiB host and GPU / 1MiB output**. No new image/weight copy.

Pinned DINOv2 source, weight, crop/context/resize/norm and causal four-field
interface stay byte-identical. Strict frozen loading, finite768-D unit features,
singleton-versus-batch tolerance1e-5, available prefix1/2 invariance to a poisoned
future tail and unchanged model state are required. No GT, roles, text or Test
input; the inference process denies their opening and network/child execution
after preflight. The private compact engineering payload cannot become formal
M2 cache. This tiny gate cannot establish coverage, association quality,
scientific PASS, frame-online decisions or primary-freeze permission. The
limited-coverage choice remains pending; no training/R2/full cache is started.
