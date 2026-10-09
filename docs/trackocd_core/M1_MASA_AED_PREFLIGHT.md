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
weight ceiling, not downloaded or locally hash-verified. Inspect the first
checkpoint's tensor coverage before deciding whether the second is necessary.
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

Next: exact binary-wheel dependency resolution and safe frozen checkpoint
key coverage, then the bounded native proposal smoke if those gates pass.
Smoke success would still require separate physical/coverage evaluation
before M1 qualification. Preserve Q0, all old streams/shards and the negative
GT representation findings. The one correction round remains consumed;
GT policy-learning scope is still awaiting explicit clarification.
