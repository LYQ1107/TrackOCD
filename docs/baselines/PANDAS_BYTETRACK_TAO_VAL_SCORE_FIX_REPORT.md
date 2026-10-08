# PANDAS → ByteTrack TAO Validation score-contract repair

Status: **COMPLETE**. This is the full TAO Validation rerun after repairing
the PANDAS-to-ByteTrack score contract and disabling the MOTChallenge
aspect-ratio filter that is not appropriate for TAO. It is a diagnostic
baseline, not a causal TrackOCD-v2 result.

## Scope and frozen inputs

- Dataset: TAO **Validation** only (`TAO_OW`, `val`, all 988 videos). TAO Test
  was not accessed or evaluated.
- No PANDAS retraining and no K=500 rediscovery were performed in this repair.
- Frozen detector: COCO-half checkpoint
  `outputs/coco_half_base_phase/model_best.pth`, saved epoch 18,
  SHA256 `86ced2e8061722aae173905585fba204c0e085f16dd2b68dec9ccfbf2344a725`.
- Frozen discovery: PANDAS K=500 prototypes
  `outputs/pandas_discovery_K500/pandas_prototypes_K500.pt`, SHA256
  `83cbe5f5e832414b4f0acd633964ade79e8d7e087923da3444ec26b9dba4a864`.
- TrackOCD branch: `codex/pandas-bytetrack-score-fix`; final implementation
  source commit `c3ace7e`. The two implementation commits are `2ea2f1e` and
  `c3ace7e`.
- Canonical TrackEval: commit
  `12c8791b303e0a0b50f753af204249e622d0281a`.

## What was repaired

PANDAS keeps the semantic/prototype score in `scores`. A separate
`foreground_scores` field is now exported as
`1 - softmax(original detector logits)[background]` and is the only score fed
to ByteTrack. `prototype_id` is retained for audit but is not used for
association. The TAO full stream is represented explicitly, including empty
frames, and the MOTChallenge aspect-ratio filter is disabled by default for
this TAO path.

On a frozen 100-image reference (30,000 detections), boxes differed by less
than `1e-5`, labels were identical, and semantic scores differed by less than
`1e-6`; semantic and foreground scores had Pearson correlation `0.344709`.
Thus the semantic detection path was preserved while the tracking score
contract was made explicit.

## Detection result

The frozen PANDAS semantic detection result is unchanged:

| Split | AP | AP50 | AP75 |
|---|---:|---:|---:|
| Base | 0.094611 | 0.152347 | 0.099922 |
| Novel | 0.002249 | 0.004533 | 0.002044 |
| All | 0.020651 | 0.033982 | 0.021545 |

These are evaluator-only Hungarian prototype-mapped detection metrics. The
full score-fix tracking cache uses a storage floor
`foreground_scores >= 0.01`; this floor is not a semantic-score replacement.

## Full-stream coverage

- 988 per-video NPZ files, 1,040,843 complete-stream frames.
- 1,039,239 frames contain retained detections and 1,604 are explicit empty
  frames.
- 162,700,608 retained detections after the foreground storage floor.
- Four shard audits are `PASS`; 296 videos were newly processed and 692
  already-valid outputs were reused.
- The full cache and logs remain outside Git. Final checked free space was
  approximately 93G.

## TAO Validation tracking metrics

All four rows use `score_key=foreground_scores`, `filter_mot_aspect=false`,
`match_thresh=0.8`, and `track_buffer=30`. Metrics below are the canonical
TrackEval combined class-agnostic TAO-OW `mean` values; MOTA is reported in
its native fraction form (so `-14.007` corresponds to `-1400.7%` in the
TrackEval display).

| Variant | track / low / new | Track rows | Tracks | Mean length | HOTA | DetA | AssA | LocA | IDF1 | MOTA | IDSW | Frag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BT-FG-0 | 0.50 / 0.10 / 0.60 | 54,461,482 | 2,370,335 | 22.9763 | 0.110113 | 0.037554 | 0.330246 | 0.811054 | 0.048112 | -14.0070 | 35,078 | 7,097 |
| BT-FG-1 | 0.20 / 0.05 / 0.20 | 65,427,220 | 3,427,596 | 19.0884 | 0.099660 | 0.032007 | 0.318292 | 0.810205 | 0.039429 | -17.2226 | 38,910 | 7,409 |
| BT-FG-2 | 0.10 / 0.01 / 0.10 | 67,455,262 | 3,640,001 | 18.5317 | 0.095504 | 0.030982 | 0.302373 | 0.809094 | 0.036867 | -17.9507 | 39,891 | 7,500 |
| BT-FG-3 | 0.05 / 0.01 / 0.05 | 65,266,712 | 3,978,336 | 16.4055 | 0.098389 | 0.031551 | 0.315260 | 0.809447 | 0.037898 | -17.5021 | 40,284 | 7,598 |

Every canonical export covered all 988 videos and all 36,375 annotated frames.
The machine-readable per-variant summaries are under
`outputs/score_fix_full/trackeval/BT-FG-{0,1,2,3}/trackeval_summary.json`.

## Old versus repaired path

The old `bf29217` direct run used semantic `scores` for tracking and kept the
MOT aspect-ratio filter enabled. It produced only 7,161 exported rows and 88
tracks, with HOTA `0.005589`, DetA `0.000217`, AssA `0.144252`, LocA
`0.831275`, IDF1 `0.000459`, MOTA `-0.001079`, IDSW 1, and Frag 1. The
repaired run is therefore a useful diagnostic comparison, but not a single
variable ablation: both the score contract and the TAO-inapplicable aspect
filter changed.

## Verification and commands

The score-contract unit test passed (`4 passed`):

```text
python -m pytest -q tests/pandas_bytetrack/test_score_contract.py
```

The final stage order was:

```text
run_pandas_anonymous_inference.py --mode full --skip-existing
run_bytetrack_npz.py --score-key foreground_scores --filter-mot-aspect=false
export_bytetrack_trackeval.py
run_tao_ow_trackeval.py --trackeval-root <pinned TrackEval> --tracker bytetrack
```

No external process was stopped or modified. Large model, data, NPZ, and
TrackEval prediction artifacts are intentionally not tracked in Git. The
complete machine-readable record is
`PANDAS_BYTETRACK_TAO_VAL_SCORE_FIX_RESULT.json`.
