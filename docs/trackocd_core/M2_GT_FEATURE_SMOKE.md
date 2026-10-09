# M2 — Common descriptor / causal-view GT feasibility smoke

2026-10-10 (Asia/Shanghai). **PASS_ENGINEERING_GT_ONLY**; not an M2 predicted
main freeze, not model training, and not a scientific hypothesis result.
M1 primary frontend qualification remains blocked. This uses the expressly
permitted GT-track representation feasibility exception, not a GT replacement
for predicted boxes in an end-to-end result.

## Pinned encoder and lawful inputs

Official [DINOv2 source](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8)
is pinned to `7764ea0f912e53c92e82eb78a2a1631e92725fc8`; the
[model card](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/MODEL_CARD.md)
specifies the ViT-B 768-D backbone. Source/license remain in a local sparse
checkout, not an embedded public Git repository. Direct official backbone
construction uses `pretrained=False`, followed by strict loading of the
already recovered 346,378,731-byte checkpoint (SHA256
`0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`).
There is no duplicate checkpoint download, classifier/text head, optimizer,
backbone fine-tuning or extra package installation. All 86,580,480 parameters
are frozen in eval mode. The absent xFormers package is not a dependency gap
for the successfully exercised official plain-tensor inference path.

Selection was registered before inference: first two sorted inherited Known
categories with qualifying long tracks in at least two Train videos, then
the first track in each of the first two videos; first 16 observations only.
This selected Known IDs **35 and 41**, **four GT tracks / 64 observations**.
This deliberately tiny long-track smoke is not a representative performance
sample. No Val/Novel/Test data or labels were read by either smoke command.

Each RGB crop uses recovered `crop_box` (10% context per side, boundary/minimum
raster behavior retained), PIL bilinear 518x518 resize, ImageNet normalization,
FP32 inference with TF32 disabled, L2-normalized `x_norm_clstoken`, FP16 storage.
The new architecture/preprocessing lineage is explicit; **byte compatibility
with old NAS features has not been established**, so they cannot be mixed.

## Actual cache and causal interface checks

One compact local cache contains `observations.npy`, `prefix_features.npy`,
`geometry.npy`, `index.parquet` and separate `train_labels.parquet`:
**138,207 payload bytes**, 142,319 bytes including completion/manifest metadata.
No crops, per-track feature JSON or full-track mean were saved. A temporary
task directory was atomically renamed only after all outputs and completion
metadata were written; existing caches are never overwritten.

`PrefixView` exposes only visible visual descriptors, normalized boxes,
quality and elapsed frames. Labels, IDs, total track length, full-track mean
and future descriptors are absent. The compact GT-only reader copies the
requested prefix directly from the read-only memmap, never loads the separate
supervision table, and explicitly rejects promotion to a predicted result.

- All five actual payload sizes/hashes verify; all four supervision keys
  align and every label is in the inherited Known whitelist.
- **20 actual views** (four tracks x p1/p2/p4/p8/p16) reproduce the saved
  FP16 means exactly: max absolute difference **0**.
- First-image descriptors extracted alone versus a four-image batch differ
  by at most **5.76e-7** in FP32; later crops are not temporal context.
- Saved unit-vector norm error is at most **3.19e-5** after FP16 quantization.
- Synthetic tests perturb every future tail for each registered prefix;
  descriptors, geometry and means remain unchanged. Model views are copies,
  not aliases to mutable full-track arrays.
- All core/recovered-v2 regression tests: **57 passed**; recovered 81-file
  NAS source identity is still tested. This is not scientific PASS.

The claim is **observation-prefix-causal**, not global frame-online semantic
decisions or persistent-memory rollouts. GT geometry is clearly labeled as
such. No detector/tracker, learned adapter/controller, formal cache, baseline
performance comparison, M9 score or M10 ablation was run.

## Resources and reproduction

One freshly idle A100 was selected by UUID after querying current compute
applications; no historical GPU number was reused. One worker / batch four;
10.78 seconds; peak allocated GPU memory **570,027,008 bytes** (~544 MiB),
peak CPU RSS **1,299,384 KiB** (~1.24 GiB). RAM plan 4 GiB, with >25% system
headroom. The sparse upstream source checkout adds **3,459,339 bytes**,
including Git metadata/cache files. Existing weight/environment/data are
reused; no OOM or external-process intervention occurred.

```bash
source /data3/liuyeqiang/trackocd_a100_env.sh
python scripts/trackocd_core/smoke_gt_features.py
python scripts/trackocd_core/verify_gt_feature_smoke.py
python -m pytest -q tests/trackocd_core tests/trackocd_v2
```

The first command requires the exact upstream checkout and checkpoint named
in `common_features.json`, and intentionally refuses to overwrite an existing
smoke. Existing valid outputs should use the second, read-only command.
Only source/config/tests and aggregate receipts are pushed; arrays, Parquet
tables, upstream Git tree and model weights remain private local assets.

Next: finish SimOWT/Q0 source/stream qualification and lawful physical-front
freeze before predicted main processing. The demonstrated GT interface may
support permitted representation feasibility; full training and formal cache
have not started and are not implied by this engineering result.
