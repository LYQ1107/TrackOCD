# PANDAS + ByteTrack on TAO Validation

Status: **COMPLETE**. This is the measured PANDAS-TAO-Val-Transductive
baseline followed by class-agnostic ByteTrack. It is not a causal TrackOCD
v2 result.

## Scope and protocol

- Dataset: TAO **Validation** only; 988 videos, 1,040,843 complete-stream
  frames, and 36,375 annotated images.
- TAO Test was not accessed or evaluated (`tao_test_used=false`).
- The base detector was trained on COCO-half, then frozen before all TAO
  discovery and inference. No TAO labels were used to train or fine-tune it.
- Discovery used image-only TAO Validation annotated-frame inputs. Category,
  box, track, and base/novel fields were not provided to discovery.
- The discovery setting was PANDAS K=500 with `invert_square`, L1
  normalization, softmax background, KMeans `n_init=10`, `max_iter=1000`,
  seed 42, 300 detections/image, and score threshold 0.
- ByteTrack used only box, score, and frame order:
  `track_thresh=0.5`, `low_thresh=0.1`, `match_thresh=0.8`,
  `track_buffer=30`.

## Frozen base checkpoint

The COCO-half base checkpoint was frozen before TAO discovery:

- checkpoint: `outputs/coco_half_base_phase/model_best.pth`
- saved epoch: 18
- SHA256: `86ced2e8061722aae173905585fba204c0e085f16dd2b68dec9ccfbf2344a725`
- PANDAS source commit: `9a6bafa728a14f3c7e936f1ee23d53110f362e85`

The official base run has completion markers for all 26 epochs and a final
epoch-25 recent checkpoint. The redirected log is missing validation print
pairs for epochs 6 and 13; those values were not reconstructed or imputed.

## Detection: frozen PANDAS inference on TAO Val

| Method | Base AP | Novel AP | All AP | Base AP50 | Novel AP50 | All AP50 |
|---|---:|---:|---:|---:|---:|---:|
| PANDAS K=500 | 0.094611 | 0.002249 | 0.020651 | 0.152347 | 0.004533 | 0.033982 |

The annotated-frame predictions were frozen before evaluator-only Hungarian
prototype mapping. Distractor categories were excluded from the Base/Novel/All
sets.

## Tracking: frozen PANDAS detections plus ByteTrack

| Method | LocA | AssocA | HOTA | IDF1 | MOTA |
|---|---:|---:|---:|---:|---:|
| PANDAS + ByteTrack | 0.831275 | 0.144252 | 0.005589 | 0.000459 | -0.001079 |

TrackEval used the pinned `TAO_OW` adapter on the `val` split in
class-agnostic mode. TETA is not reported because this local TrackEval adapter
does not expose a TETA metric.

## Coverage and artifact policy

- Frozen full-stream anonymous inference: 988 per-video outputs,
  1,040,843 frames, 312,252,900 detections.
- ByteTrack: 988 videos, 7,161 track rows, 88 tracks in the exported sparse
  evaluation stream.
- Discovery: 2,265,697 features, 500 novel prototypes.
- The large model, TAO data, NPZ predictions, feature caches, and logs are
  intentionally not tracked in Git. Their run-local hashes and detailed audit
  files remain in the experiment worktree.

The machine-readable summary is
`PANDAS_BYTETRACK_TAO_VAL_RESULT.json`. The ByteTrack implementation used by
this result is `src/iclr27_phase3b/bytetrack.py`.
