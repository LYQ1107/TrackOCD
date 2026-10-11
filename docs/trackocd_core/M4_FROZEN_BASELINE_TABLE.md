# Frozen M4 baseline handoff

Actual full limited Native MASA p16 baseline mean ± four-order population SD.
Raw baselines have no training-seed repeat. Same input/features/evaluator and
fixed full GT denominators; aliases not independent. All five prefix values
in BASELINE_COMPARISON.csv; controlled Train table separately in
TRAIN_BASELINE_COMPARISON.csv. PHE INCOMPARABLE (48/78 support and common-feature
checkpoint supervision compatibility unproven); no surrogate result.

| Method | Old % | New % | actual mean H % | pure CT % |
|---|---:|---:|---:|---:|
| B0_frame_snapshot_vote | 0.6571 ± 0.0833 | 14.6215 ± 0.3385 | 1.2567 ± 0.1509 | 0.0000 ± 0.0000 |
| B1_track_nearest | 0.0227 ± 0.0160 | 13.4310 ± 0.2284 | 0.0452 ± 0.0319 | 0.0000 ± 0.0000 |
| B2_track_dpmeans | 1.7562 ± 0.0793 | 11.2943 ± 0.0611 | 3.0388 ± 0.1164 | 0.0473 ± 0.0819 |
