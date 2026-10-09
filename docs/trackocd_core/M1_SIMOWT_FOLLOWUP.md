# M1 follow-up — SimOWT assets found; no clean primary freeze

2026-10-10 (Asia/Shanghai). Follow-up to the delivered M1 diagnostics and
M2 GT-only smoke. NAS turn `01a121db-ff96-72c0-8300-04e0751c6079` completed;
it is not a live inference/extractor job. Relevant command outputs, actual
forward/mapper/tracker code and hashes were read by this A100 agent.

## What is now established

The normalized Val physical JSONL exists on NAS: **329,292,704 bytes**, SHA256
`60be1313b781e5b8809b1ad2d9d75e303e0ec3b90664cdd7be1f8d281fd95038`.
A bounded source-side streaming scan independently counts **988 videos,
649,378 tracklets / 1,853,369 observations**, with zero array-length mismatch.
All rows are `val_predicted`; no category, text or embedding is exported.

- Length mean **2.8541**; p25/median/p75 **1/1/2**, p90/p95/p99 **6/11/30**.
- **406,425 single-observation tracklets (62.5868%)**. This is the actual
  highly fragmented ~650k route; do not confuse it with PANDAS median 10.
- Historical normalization retained **16,290 degenerate boxes**, rather than
  silently dropping difficult observations. The subsequent native-row scan
  independently recounted this same degenerate-box total.
- Original public, early-v2, normalized physical and native evaluator streams
  have distinct sizes/hashes/schemas. They are not interchangeable assets.

The full fixed non-distractor GT universe and historical temporal-IoU score
keys both contain **5,232** tracks, no missing/extra keys. Recounting those
saved scores yields Known **1618/4413 (36.66%)** and Novel **179/819 (21.86%)**
at the inherited geometry-only >=0.5 threshold. Geometry matching itself was
not rerun. Historical persistent observability **83/527**, target observability
**103/527** are preserved diagnostics, **not Correct Commit-CT**.

Preserved full-Val TrackEval (0–1 scale): HOTA **.15279**, AssA **.48950**,
DetA **.048456**, DetRe **.75133**, DetPr **.049034**, OWTA **.60202**.
The TrackEval projection includes 5,485 GT IDs including distractors and
599,375 predicted IDs; these are not the raw normalized-stream counts or the
5,232-target TrackOCD denominator. No IDF1 number is fabricated.

Four exact small source files were restored privately, **36,132 bytes**:
frontend stage JSON, normalization JSON, download manifest and current full
tracker source. All destination size/SHA256 match NAS. Full metric JSON,
large streams, weights, raw log and private labels were not transferred.
Curated facts/hashes are in `nas_simowt_provenance.json`; original `clean=true`
declarations are preserved privately but not accepted as proof.

## Legality, provenance and time boundary

The old log confirms `--eval-only`, a one-class classifier, `COCO_PRETRAIN=True`
and 36,375 TAO Val images. The registered COCO-style dataset name does not
mean the inference dataset was COCO. Config text matches the log after one
trailing empty line; raw snapshot hashes differ by that newline.

The present 542,162,424-byte checkpoint SHA256 matches the download manifest:
`70f30d3b66684e58d1875c75ba1c1011984644ea8362e7761dbbb6366a392a72`.
The [official model zoo](https://github.com/22109095/SimOWT/blob/753b2ac0082976753350d1b74a4d332088889e57/assets/train%26test%26model_zoo.md)
points to the same Drive ID. Its [training configuration](https://github.com/22109095/SimOWT/blob/753b2ac0082976753350d1b74a4d332088889e57/projects/IDOL/configs/r50_train.yaml)
names class-agnostic COCO training, but does not establish every training
stage of the released checkpoint. Complete exclusion of Novel/Val label
supervision remains **UNVERIFIED**, not proven leakage.

No Novel text vocabulary is consumed by the inspected inference branch.
However, the old evaluator passes the complete Val dataset, including
annotations/future records, to the model entrance. The actual eval branch
discards that second argument; the mapper removes per-image annotations.
The constructor reads the annotation JSON but retains video lookup records.
Static association uses current predictions plus past memo (10 frames;
three long embeddings); no future geometry/ID rewrite was found. This is
not a runtime proof of strict input isolation or frame-online replay.

The old SimOWT repository was dirty, and the run did not record a weight
SHA256 or a sealed source patch. Two stage/input hash references differ from
current files. That prevents a complete historical binding; it does not
alone prove corrupted results. Old-root dependency links remain broken.
The old Python 3.7 / torch 1.8 runtime was not installed on A100.

## Reproducible score-contract diagnosis, not a repair

Source inspection shows `coco_inference` returns probability-valued scores,
but `track_eval` applies another sigmoid. Values in [0,1] become
**[0.5, 0.731059]**, so selection 0.1 and add-new 0.2 cannot reject a low raw
probability at these gates. NMS, masks and association still operate.

The empty-memo branch also uses a shared view of the box score column and
assigns 0.7501 through that view before initialization. A CPU diagnostic
executes only three AST-verified source statements, with no upstream import,
checkpoint, data or GPU. Synthetic scores [.01, .19, .9] become [.7501]*3:
initialization at .2 accepts **3 instead of 1**. The same operations are
present in pinned upstream source; they were not invented by the recovery.

This proves two static contract effects. The following additional scan also
measures the actual saved score distribution. Neither alone proves exact
historical code/weight binding or all fragmentation's cause.
No frozen scores/boxes/IDs were modified, no threshold sweep or full inference
was run. Any later correction would require its own distinct artifact,
bounded smoke and evaluation; it cannot be passed off as original Q0.

## Decision and minimal next assets

**M1 remains BLOCKED_FRONTEND_QUALITY; selected primary = none.** SimOWT is
an inspected candidate/reference with unresolved supervision/runtime binding
and a diagnosed score contract. Unlike COVTrack-native, it has **no positive
evidence of Novel vocabulary use**. Do not conflate UNVERIFIED with that
candidate's FAIL_NO_NOVEL_VOCABULARY.

Before any primary freeze, resolve supervision provenance and use a scoped,
hash-verified stream transfer for independent geometry/score validation.
Only the normalized physical + native evaluator JSONL would be necessary:
**624,028,441 bytes (~595 MiB)**. The 1.22-GB raw merge, duplicate public/early
streams, TAO frames and old environment are unnecessary for that audit.
The checkpoint is conditional on a separately justified bounded rerun.
No bulk transfer or full feature cache is automatically authorized.

## Additional actual-stream score check

NAS turn `01a1220d-d968-7153-884c-69e70550ec99` completed the user-approved
read-only native evaluator JSONL scan. Main agent read the full command and
untruncated output. File size/SHA256 match; size/inode/mtime unchanged.

- **1,853,369 observations / 988 videos**, no blank/malformed rows.
- Score min **.5053257942**, max **.7501000166**, mean **.6140423702**.
- **1,802,527** scores in [0.5, sigmoid(1)] and **50,842** near .7501,
  with endpoint/absolute tolerance 1e-6. These disjoint sets contain every
  score; **zero below .5** or outside both sets, zero nonfinite/missing scores.
- All `category_id=1`; **16,290 degenerate finite boxes**, no malformed or
  nonfinite boxes. No new annotation/GT/Test read was needed.
- One worker, **28.664 s**, **19,345,408-byte peak RSS**; enforced 256-MiB
  address-space cap, minimum sampled system available RAM **81.85%**.

This is strong artifact-level consistency with the inspected recompression
and overwrite operations, not mere synthetic algebra. It does not recover
lost raw probabilities or prove the checkpoint's supervision history. No
original score/box/ID was changed; no inference/caching job started.
The aggregate receipt is `nas_simowt_score_distribution.json`. The primary
frontend remains unqualified. The latest regression including this receipt
is **111 passed**; earlier 62 refers to the original follow-up delivery.

The explicit GT-feasibility exception remains available. M2's four-track,
64-observation frozen DINO smoke is only PASS_ENGINEERING_GT_ONLY; it cannot
replace a qualified predicted-track frontend or the later M9 main results.

## Verification

**62 tests passed** including the 81-source-file byte-identity regression.
The first restricted-builtins CPU run failed because torch's lazy CPU path
requires `__import__`; supplying it only after exact AST verification fixed
the diagnostic without importing the upstream tracker or running any model.
The numerical receipt is `simowt_score_contract.json`.

```bash
source /data3/liuyeqiang/trackocd_a100_env.sh
python scripts/trackocd_core/audit_simowt_score_contract.py
python scripts/trackocd_core/audit_frontend.py --refresh-candidates-only
python -m pytest -q tests/trackocd_core tests/trackocd_v2
```

The CLIs require the private, hash-pinned source copies; synthetic tests do not.
The metadata-only refresh preserves original physical counts/metrics and
their timestamp; it is explicitly marked as not rerunning those diagnostics.
Private copies/raw logs/annotations are excluded from Git. This follow-up
created only ~51 KiB of private source/aggregate receipts plus small code/docs.
Disk recheck: 94,587,236,352 bytes available; RAM 108,182,560,768 available.
No TAO Test data/annotation access or external-process interference occurred.
