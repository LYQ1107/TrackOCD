# Frozen PANDAS / ByteTrack diagnostic reproduction

Run commands from the repository root using the existing Python environment.
`source_manifest.json` pins the upstream PANDAS and canonical TrackEval commits
and records SHA256 for the actual preceding experiment entry points.
`frozen_runtime/` contains those small source snapshots; it is not an instruction
to repeat GPU inference, training, or discovery. The foreground export change is
retained in `patches/pandas_foreground_score.patch`, applicable to PANDAS commit
`9a6bafa728a14f3c7e936f1ee23d53110f362e85`.

The diagnostic reads one frozen video at a time. Original files under
`outputs/anonymous/full` are used only to count pre-floor detections and audit
pre-floor proposal recall. The tracking-only path reads
`outputs/score_fix_full/anonymous/full`; original FG-0 tracks are reused from
`outputs/score_fix_full/bytetrack/BT-FG-0`.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /data3/liuyeqiang/.venvs/trackocd-a100/bin/python \
  scripts/baselines/pandas_bytetrack/audit_redundancy.py \
  --asset-root /data3/liuyeqiang/pandas_bytetrack_tao_val \
  --annotation /data3/liuyeqiang/TAO-Amodal/annotations/validation.json \
  --output-root outputs/baselines/pandas_dedup
```

The first 20 ascending IDs are the calibration videos. A disjoint confirmation
set is fixed by SHA256 of `pandas-dedup-confirm-v1:<video_id>` before reading
metrics. `selection.json` records those sets and the three allowed NMS variants.
No ByteTrack thresholds are selected in this diagnostic.

Canonical TrackEval evaluates only TAO image-table frames after association has
seen the complete stream. With `SUBSET=all`, all GT annotations (including
distractors) are collapsed to object. Its unmatched-object filtering checks
category placeholder 1 against the original video flags; category 1 is absent
from both negative and nonexhaustive flags on all 988 Val videos. Therefore
unmatched predictions are ignored on GT-empty frames and counted on frames
containing GT. They may include unannotated real objects; reported unmatched
counts must not be interpreted as a verified false-object count. This pinned
behavior is identical for the incumbent and new variants.

The new output root has a 10 GiB soft budget. Models, data, source NPZ and old
tracking outputs remain read-only and outside Git.

## Tracking adapter and tests

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /data3/liuyeqiang/.venvs/trackocd-a100/bin/python -m pytest -q \
  tests/baselines/pandas_bytetrack/test_tracking_postprocess.py

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /data3/liuyeqiang/.venvs/trackocd-a100/bin/python \
  scripts/baselines/pandas_bytetrack/run_tracking.py \
  --detections-root /data3/liuyeqiang/pandas_bytetrack_tao_val/outputs/score_fix_full/anonymous/full \
  --output-root outputs/baselines/pandas_dedup/calibration/Dedup-A/tracks \
  --budget-root outputs/baselines/pandas_dedup \
  --selection outputs/baselines/pandas_dedup/selection.json \
  --subset calibration --variant Dedup-A
```

Use the same runner for the preregistered B and C variants with their own output
roots. It never loads annotations. It retains original detection indices and
all frame/image IDs. Original offsets are preserved as `source_frame_offsets`;
suppressed candidate/track arrays have new cardinalities, so their own offsets
must change. Resume validates input/output SHA256 and exact NMS configuration.
