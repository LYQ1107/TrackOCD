# M1 — Diagnostics complete; physical frontend qualification blocked

2026-10-10 (Asia/Shanghai). Exact v2 recovery was committed/pushed as
`ccbdb0696175cfb0bbb8134ed193038f6905c7d8` and independently verified at the
remote HEAD before these M1 diagnostics. The inherited 81 source files remain
byte-identical. No new detector/tracker training or inference was performed.

## Actual frozen PANDAS → ByteTrack reference

The existing BT-FG-0 foreground-score stream is read-only; no extra threshold
search, dedup change, inference or training is authorized by this audit.
`audit_frontend.py` reads only frame ordinals/offsets/physical track IDs in all
988 NPZ files and hashes the existing canonical prediction JSON by streaming
bytes. The hash equals its export receipt:
`245c278e163d431ccf2bc8dd5881f621773a4fd11b44ef8a089ed8aa3ec60e7a`.

- Full association stream: 1,040,843 frames, **2,370,335 physical tracks**,
  **54,461,482 observation rows**; all counts match its frozen runtime receipt.
- Mean track length 22.9763; p10/p25/median/p75/p90 = **1/4/10/24/52**.
- Single-observation tracks: **241,433 (10.1856%)**; length <=4:
  **29.3757%**; length <=16: **64.6164%**.
- Do not label the full stream "mostly single-frame". Long tracks and an
  enormous unmatched population coexist; average length does not establish
  reliable Novel target coverage.

Existing, independently hash-linked full-Val class-agnostic TrackEval metrics
(0–1 scale): HOTA **0.110113**, AssA **0.330246**, DetA **0.037554**, mean
DetRe **0.578390**, IDF1 **0.048112**. These are frozen reference metrics,
not a new experiment. The old Dedup-C subset result is not used.

## New evaluator-only coverage diagnostic

`audit_frontend_coverage.py` evaluates all inherited non-distractor Val GT
tracks. Geometric matching uses no category input: one per-video Hungarian
and the already registered legacy temporal-IoU >=0.5 rule. Labels aggregate
Known/Novel results **after** geometry matching. Temporal IoU is sum of
same-image box IoUs divided by the union of canonical image-ID frame sets.

- Known: **1,442 / 4,413 = 32.6762%** reliably observed; 2,971 missing/unreliable.
- Novel: **30 / 819 = 3.6630%** reliably observed; 789 missing/unreliable.
- Full fixed non-distractor GT universe: **5,232** tracks. Missing GT is not
  dropped from either coverage denominator.
- Canonical projection: **36,375 annotated frames / 1,831,391 predicted rows**,
  agreeing with the frozen TrackEval export counts. The original association
  processed intervening frames, while geometric scoring is on the canonical
  annotation-frame universe shared with the legacy frontend comparison.
- This diagnostic does not calculate matched-only OCD accuracy, update model
  memory, tune a threshold or replace the later M3/M9 evaluator.

Projection count agreement is **not** byte identity of all NPZ boxes versus
the export, and full NPZ checksums were not computed. The canonical exported
prediction hash was independently checked by the separate length audit.
These limitations remain explicit because no frontend freeze is claimed.

Both diagnostics ran one CPU worker, no GPU, with <2-GiB working plans and
more than 25% system RAM headroom. Existing ~1.10-GiB NPZ inputs were reused,
not duplicated. Only small aggregate evidence JSON files were created.

## Recovery candidate and qualification decision

COVTrack-native has a located NAS stream (369,794 tracks / 1,133,841 rows)
and 255/723 partial shards, but no stream/shard was transferred. The NAS
Val-only audit completed in turn `01a121c9-3912-7a43-ac61-e4f0b131c1b4`.
Five original audit JSON files (260,702 bytes total) were reconstructed from
complete source-side chunks into an ignored private evidence directory.
Every A100 size and SHA256 matches NAS; original files/claims remain unchanged.
Curated reproducible facts and source command references are recorded in
`outputs/trackocd_core/audit/nas_m1_provenance.json`, without publishing the
internal research log, AGENTS, vocabulary strings or raw Val annotations.

The current native config uses 1,203 LVIS class text embeddings and
`only_validation_categories=True`. Source code selects a **296-class Val
vocabulary**, computes detection scores by multiplying visual region
projections by the text matrix, and uses a Val Novel mask to choose blending
weights. Evaluator-only **v0.5 -> v1 synset** mapping finds **203/209 TrackOCD
Novel classes in that selected vocabulary**; six have no v1 mapping. Numeric
IDs from the different vocabulary versions were not compared directly.
The historical run's initialization log independently confirms loading
`detpro_prompt.pt`, `custom_classes=False` and text shape **[296, 512]**.
Its observed 295 prediction categories all lie in the current selected list.
Thus this is not merely a dormant source-code path.

`confused_features=True` additionally fuses appearance/location/visual
semantic embeddings before association. The semantic embeddings are visual
text/image projections, not direct passage of the text matrix to the tracker;
`cls_static_ratio=0` does not disable this fusion. Removing output categories
cannot undo vocabulary-dependent detection selection and confidence.
**Native legal gate: FAIL_NO_NOVEL_VOCABULARY; core comparability:
INCOMPARABLE_VOCABULARY_ASSISTED_REFERENCE.** COVTrack-NoSemantic disables
association fusion only, so it does not repair the shared detector contract.

Preserved historical native metrics (not recomputed here): HOTA 0.15412,
AssA 0.44569, DetA 0.055848, DetRe 0.54448; Known coverage **1015/4413**,
Novel coverage **191/819**, Persistent Observability **88/527**. These are
frontend diagnostics, not downstream OCD/Commit-CT or lawful main results.

Static detection/association uses current image and historical memo (10
frames); no future geometry rewrite was found. Offline format-time majority
vote does use future categories, but modifies **only category_id**, not
boxes/scores/physical IDs. This fact must not be mislabeled as proven future
box/ID smoothing. Runtime frame-online replay is not verified. Full training
supervision exclusion is also unverified; "C-TAO base" is not the inherited
TrackOCD Known split.

Lineage limitations remain: the old repository was dirty; native run saved
neither config nor prompt SHA256. Two historical stage/input references have
different hashes from the present files. The missing input snapshots prevent
a fully verified historical configuration chain, but do not by themselves
prove corrupt artifacts. Current NAS dependency links to old `/data1/...`
are broken; candidate files elsewhere are not assumed to resolve those links.

OVTR-native is explicitly vocabulary-assisted/reference-only: **INCOMPARABLE**
as a clean core frontend. The subsequent
[SimOWT follow-up](M1_SIMOWT_FOLLOWUP.md) verifies its source-side stream hash,
649,378-track fragmentation profile and actual static inference branch.
Checkpoint supervision/historical runtime binding remain unverified and
score-contract defects are diagnosed; it is not frozen as clean primary.
No existing lawful MASA/AED TAO stream is established. The subsequent
[MASA and AED preflight](M1_MASA_AED_PREFLIGHT.md) identifies a conditional
MASA-SAM-B native-proposal smoke, not a qualified frontend. Its default TAO
configuration depends on external Detic detections; a frozen image-only
RPN/ROI wrapper, compatible runtime and release tensor coverage are unverified.

**M1 status: diagnostics complete / BLOCKED_FRONTEND_QUALITY.** No qualifying
main frontend has been frozen; `FROZEN_PHYSICAL_FRONTEND.json` is not written.
PANDAS remains a fixed external reference, not promoted to primary by this
audit. The goal remains active. The user-authorized **GT-track feasibility
exception** authorizes representation feasibility, clearly labeled GT;
GT policy learning is awaiting explicit scope clarification. Neither can
replace the required predicted-track M9 main result or prove overall
completion. The subsequent M2 GT-only smoke completed four tracks /
64 frozen DINO observations, without training or a predicted main result.

The verified live NAS metadata audit was waited on until completion, not an
old source-upload handle. The narrow SimOWT/Q0 follow-up has also completed;
neither completed audit is a live extractor. The original 81 source files,
255 source cache units and all historical results are preserved.

## Verification and delivery

All core and recovered-v2 tests pass: **55 passed**. The tests include source
identity, video-local IDs, fixed temporal-IoU matching and one-to-one coverage;
they are engineering checks, not scientific PASS. `git diff --check` passes.
The two read-only diagnostics use the existing A100 environment:

```bash
source /data3/liuyeqiang/trackocd_a100_env.sh
python scripts/trackocd_core/audit_frontend_coverage.py
python scripts/trackocd_core/audit_frontend.py
python -m pytest -q tests/trackocd_core tests/trackocd_v2
```

Full diagnostic reruns require existing Val/BT-FG-0 assets and the five private
metadata copies named in the curated provenance receipt. Only aggregate
reports, receipts, scripts and synthetic tests are delivered to GitHub;
private source copies are deliberately excluded.

## Latest separate MASA all-Val physical audit

The previously unverified candidate is now restored and executed under separate
remote preregistration455e9dd; sealed inference delivered414ebda. Exact pinned
source/frozen419-tensor model, existing images, video-atomic output hashes and
causal image-only wrapper are verified. Published generic-image provenance is
supported, but exact private SAM release-stage ledger is not claimed.

Actual full988-video/36375-image result: MASA HOTA0.145054/AssA0.440879,
Known1400/4413/Novel189/819; PANDAS frozen reference HOTA0.110113/AssA0.330246,
Known1442/4413/Novel30/819. PANDAS HOTA recomputes exactly (difference0).
Native Novel coverage23.08% is6.3x PANDAS, yet630Novel targets missing/unreliable;
median native length2, singletons37.56%, unknown rows1456292/1540022. Neither
proposal counts nor matched-only purity establish a strong frontend or M9 PASS.
See `M1_MASA_AED_PREFLIGHT.md` and `masa_full_val_physical_result.json` for all
metrics, denominators, per-video records, resources and cadence limitations.
Primary remains unfrozen pending an explicit quality/limited-coverage choice;
no post-Val threshold/cap/weight change or downstream semantic training follows.
