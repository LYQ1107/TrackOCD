# PANDAS → ByteTrack TAO Val score-contract repair

Status: **COMPLETE**. Full TAO Validation inference, four threshold sweeps,
canonical TrackEval export/evaluation, and final result recording are done.

This progress record belongs to the `codex/pandas-bytetrack-score-fix`
branch, based on `bf29217`.

## Completed gates

- Frozen base checkpoint and K=500 PANDAS prototypes were reused; no training
  or discovery was rerun.
- A 100-image annotated-frame reference was frozen before the patch. After the
  patch, boxes were identical within `1e-5`, labels were identical, and the
  maximum semantic-score difference was below `1e-6`.
- PANDAS now exports a separate `foreground_scores` field computed as
  `1 - softmax(original_COCO_detector_logits)[background]`. The legacy
  `scores` field remains the PANDAS semantic/prototype score.
- The 100-image contract smoke passed: 30,000 detections, semantic/foreground
  Pearson correlation `0.344709`, and the fields were not equal.
- Wide-object and empty-frame tracker tests passed. A four-frame NPZ runner
  smoke also passed with two explicit empty frames.

## Fixed 20-video smoke

The first 20 TAO Validation videos were run with a storage floor of
`foreground_score >= 0.01`. The export covered 22,322 frames and retained
3,736,349 detections after the floor; all frame offsets were present and no
frame was empty after the floor in this fixed subset.

All four threshold configurations used the foreground score and disabled the
MOTChallenge aspect-ratio filter:

| Variant | track | low | new-track | Track rows | Tracks | Mean length |
|---|---:|---:|---:|---:|---:|---:|
| BT-FG-0 | 0.50 | 0.10 | 0.60 | 1,109,433 | 49,367 | 22.4732 |
| BT-FG-1 | 0.20 | 0.05 | 0.20 | 1,360,479 | 73,656 | 18.4707 |
| BT-FG-2 | 0.10 | 0.01 | 0.10 | 1,406,559 | 79,945 | 17.5941 |
| BT-FG-3 | 0.05 | 0.01 | 0.05 | 1,353,202 | 87,889 | 15.3967 |

This is threshold sensitivity evidence only; it is not a Val-selected final
operating point. The full TAO Validation completion is recorded below.

## Full TAO Validation completion

The full rerun reused the frozen COCO-half checkpoint and K=500 prototypes.
All 988 videos and 1,040,843 complete-stream frames were covered; the cache
contains 162,700,608 detections after `foreground_scores >= 0.01`, with 1,604
explicit empty frames. All four canonical TrackEval exports and all four
TAO-OW evaluations returned `PASS` over 36,375 annotated frames. The final
metrics and hashes are recorded in
`PANDAS_BYTETRACK_TAO_VAL_SCORE_FIX_REPORT.md` and
`PANDAS_BYTETRACK_TAO_VAL_SCORE_FIX_RESULT.json`.

Large models, data, NPZ outputs, and logs remain outside Git.
