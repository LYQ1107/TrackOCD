# Train nested-GT metadata access incident

Recorded 2026-10-12 Asia/Shanghai, before any new training (0/600 GPU-hours).
This is an access-boundary correction, **not a clean-access PASS** or model
result. No new scientific job is launched during containment.

## What was wrong

The original `image_video_metadata()` skipped the top-level `annotations` and
`categories` values but called `json.loads` on the entire `images` and `videos`
arrays. Train video objects include `neg_category_ids` and
`not_exhaustive_category_ids`: these are GT-derived metadata, not merely paths
or chronological indices. They entered CPU memory before the caller selected
its permitted videos. Earlier blanket statements that no labels were decoded
were too broad and must not be used as scientific access certification.

A field-name-only byte search found 500 occurrences of each field, with 492
and 367 nonempty numeric arrays respectively. No category IDs/names, video
memberships or per-row values were printed or copied in that check. This is
not a new full-GT or heldout category census.

## Preserved results and narrowed interpretation

The immutable incident receipt is
`outputs/trackocd_cvpr/safety/train_nested_metadata_access_incident_v1.json`.
It binds the original preflight and pixel-stat result hashes and their exact
historical reader bytes. Neither original result was rewritten. Pixel counts,
file sizes and existing valid caches are retained; **they are not certified
lawful unseen-class evidence by this remediation**.

The preflight's `final_holdout_labels_read=false` is not a blanket no-GT-access
guarantee. The pixel inventory's `old_final_heldout_labels_decoded=false`
describes only the explicit target-label Parquet predicate; it says nothing
about the video GT arrays decoded from the whole Train JSON. Similarly,
`annotation_or_category_arrays_JSON_decoded=false` is only accurate for the
two top-level arrays, not every category-bearing field.

`metadata_universe()` produces only video/image IDs, chronology and paths;
`validate_plan()` checks exact sanitized plan field sets; the inspected MASA
input path constructs current RGB/geometry/ordinal metadata, not these GT
arrays. This is scoped source evidence against forwarding the offending
fields, **not a proof of every historical runtime's behavior** and not an undo
of the access. Dependent diagnostics therefore retain an unresolved access
attestation overlay, rather than being discarded or relabeled successful.

## Containment

The replacement reader projects individual image/video objects **before
JSON-decoding their values**. Images permit only `id`, `video_id`,
`frame_index`, `file_name`, `width`, `height`; videos permit only `id`.
Unknown/nested values are syntax-scanned as bytes, never JSON-decoded or
returned. Keys are decoded to identify fields; no whole metadata object or
array is passed to `json.loads`. Non-scalar allowlisted values, wrong scalar
types, negative indices, malformed separators/escapes and excessive nesting
are rejected. This is code hardening, not a retroactive access repair.

Synthetic decode instrumentation checks that actual forbidden sentinels and
nested numeric GT arrays never reach `json.loads`. Nested-GT value mutations
must leave the projected metadata and sanitized physical plan unchanged.
The source-header reader still permits only plan `sources` and annotation
SHA; private plan rows/summary remain skipped.

No real annotation read using the corrected reader is claimed yet. No new
GT target-label read, training, model selection, pixel decoding, data/weight
transfer, full cache job or Val/Test access occurred during this containment.
Original checkpoints, shards, ledgers, manifests and other projects remain
untouched. The separate, earlier verified two-ZIP cleanup is not remediation
for this incident and is not repeated.

## Unresolved gate

New scientific execution remains gated until the affected access scope and
fresh lawful protocol are resolved. Synthetic tests and source projection
alone cannot certify a blind final holdout. Do not reset the old history,
reuse old adapted models as unseen-class baselines, select a convenient new
holdout from observed labels or call this goal complete. Source-only safety
repair and audit may proceed; no authorization to widen GT access is inferred.
