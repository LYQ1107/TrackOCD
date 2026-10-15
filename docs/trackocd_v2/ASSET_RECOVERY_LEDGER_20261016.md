# TrackOCD v2 minimum asset-recovery ledger — 2026-10-16

This ledger separates what is actually present on the new server from what is
only a historical report. It is an admission/recovery plan, not a formal
benchmark result. No TAO Test asset is included.

## Current machine

| class | current evidence | handling |
|---|---|---|
| v2 source | branch `research/cvpr-trackocd-autonomous-v2`, HEAD `7d3fcc5` | present; preserve and verify remote before every run |
| Python/CUDA environment | `/data3/liuyeqiang/.venvs/trackocd-a100`, entered through `trackocd_a100_env.sh` | present and reusable; add dependencies only after an import gap |
| repository footprint | about 5.0G including local outputs; tracked source itself is about 17MB in the takeover record | do not duplicate the repository or historical outputs |
| free space | `/data3` 1.7T, `/data2` 1.8T, `/data1` 3.3T | enough for a staged recovery, but no blind cache/data copy |
| TAO Train/Val data | `/data3/liuyeqiang/TAO-Amodal`, about 358G; exact Train/Val annotation hashes verified | now exposed through reversible symlinks only; never copied or Test-mapped |

## Required formal-v2 inputs

| class | exact expected asset | current status | recovery rule |
|---|---|---|---|
| protocol roles | `data/tao_ow_ocd_v1/splits/{known_ids,unknown_ids_val,distractor_ids}.json` | present as symlinks to byte-recovered files; hashes/counts `78/209/45` verified | keep the mapping manifest; do not recreate from metrics |
| Val annotations | `data/raw/tao/annotations/{train,validation}.json` | present as symlinks to the hash-verified TAO-Amodal files | use only Train/Val paths; admission never parses these large files |
| public physical stream | `data/tao_ow_ocd_v1/public/pred_track_stream.jsonl` | missing locally; later historical NAS audit records 258,036,479 bytes, SHA256 `63945d2ae19f...`, 369,794 tracks and 1,133,841 observations | recover exact file and verify the declared hash/row contract; do not synthesize from counts |
| frontend selection | `outputs/trackocd_v2/audit/frontend_selection.json` | missing in the formal namespace | recover with provenance; historical core copy is evidence only |
| representation decision | `outputs/trackocd_v2/audit/representation_decision.json` | missing in the formal namespace | recover and verify final status/authorization |
| COVTrack-native stage | `outputs/trackocd_v2/audit/frontend_covtrack_native.json` | missing in the formal namespace | require same-v2 physical/evaluator contract and `test_semantic_accessed=false` |
| formal Val cache | `outputs/trackocd_v2/features/formal/<selected_frontend>/cache_manifest.json` | missing | accept only `status=COMPLETE`, Val split, complete `.done` markers and all listed shard files |

## Historical or optional assets — not formal inputs yet

| class | evidence | status/space policy |
|---|---|---|
| COVTrack-native audit | `outputs/trackocd_core/audit/nas_m1_source/` (five small JSON files, about 250KB total) | preserved historical Val evidence; explicitly `HISTORICAL_ASSET_AUDIT_ONLY` and `same_v2_predicted_stream=false`; retain in place |
| native checkpoint | historical NAS audit records `ctao_public.pth`, 368,829,353 bytes, SHA256 `e4d0b65798844280ea13943e580ce6233ae5d331c4aa1d38eb226c3208dd567c` | actual file absent locally; recover only with matching config/output hashes and a fresh legality audit |
| old formal shards | later historical NAS audit records `255/723` shards, `130560/369794` tracks, `561672/1133841` observations, 35.3061% coverage and 2,087,621,953 payload bytes | incomplete historical snapshot; the earlier 65-shard count is obsolete evidence, but neither snapshot is current until NAS recheck |
| DINOv2 source/weights | local checkpoint `checkpoints/dinov2_vitb14_pretrain.pth` is 346,378,731 bytes with SHA256 `0b8b82f85de9...`; vendor source `vendor/trackocd_core/dinov2_7764ea0` is commit `7764ea0...` with 156 Python files | present and hash-verified; do not copy/download again; this does not replace the missing COVTrack frontend checkpoint |
| PHE/H3 artifacts | expected PHE checkpoint and Phase90 H3 namespace absent | recover exact files; never replace H3 with a Phase89/90 diagnostic |
| independent pseudo-Novel Train cache | `outputs/trackocd_cvpr/cvpr2/independent_GT_video_cache_v1` result, 68 videos/295 tracks/7,103 observations | present as anonymous GT-control preparation only; not predicted stream, not formal OCD/MOT, not a model-selection input |
| local historical MASA physical stream | `outputs/trackocd_core/physical/masa_full_val_annotated/` | 988 Val video shards, 36,375 annotated images, 1,811,677 detection rows and 1,540,022 prediction rows pass a new read-only integrity audit; annotated-cadence and provenance-gapped, so quarantined as physical evidence and not admitted to v2 |

## Safe recovery order

1. Restore a route to the NAS or locate an already-mounted backup. Inventory
   directory entries, symlink targets, size, mtime, SHA256, manifest status and
   completion markers before copying any large object.
2. Recover small code/protocol/audit files first. Run
   `scripts/trackocd_v2/recovery_asset_admission.py` with explicit data and
   output roots; it must report `READY_FORMAL_V2_PREDICTED_EVAL_INPUTS`.
3. Recover only the valid predicted stream and complete selected-frontend
   shards. Keep incomplete/failed shards quarantined and reuse complete shards
   only after their input and source hashes match.
4. Run the frozen predicted evaluator, then the causal OCD/MOT evaluation and
   reports. No training, Test read, or full cache launch is authorized merely
   by finding a historical metric or the old 65-shard count.

The current admission result is therefore **blocked on exact predicted-stream,
frontend-provenance and formal-cache recovery**, while the research Goal
remains active and the historical evidence remains preserved. The applied
Train/Val-only symlink mapping is recorded at
`outputs/trackocd_cvpr/v2_local_reversible_mapping_20261016.json`.

## Historical NAS snapshot boundary

`outputs/trackocd_core/audit/nas_source_audit.json` is a local record of a
previous read-only NAS inventory, not a live mount. It is useful for locating
and verifying future recovery, but its limitations explicitly include no
per-file checksum/Parquet-row/numerical validation, no proof that NAS source
equals the current Git commit, and no A100 restoration. The current server
must re-open the actual files before any shard reuse.

The DINO common encoder is a separate positive recovery result: its local
checkpoint and pinned vendor source match the historical hashes. It is safe to
reuse for a future registered cache smoke, but it does not make the missing
COVTrack-native frontend or predicted physical stream available.

## Local MASA migration eligibility audit — 2026-10-16

The existing local MASA output was checked without opening annotations, GT or
role files, images, Test assets, or model weights. The audit script is
`scripts/trackocd_v2/audit_masa_frontend_migration.py`, and its small report is
`outputs/trackocd_cvpr/v2/masa_frontend_migration_eligibility_v1.json`.

- All `988/988` NPZ shards and `988/988` completion receipts were present and
  mutually consistent. The one-shard-at-a-time checks passed finite box/score,
  shape, offset, receipt byte-count and SHA256 validation.
- The sealed manifest reports `36,375` annotated images, `1,811,677`
  detections and `1,540,022` physical prediction rows. The payload exposes
  image/frame indices, track IDs, boxes and scores, with no category/text/GT
  field observed.
- The manifest explicitly records `test_access=false`,
  `gt_or_role_model_input=false`, and `training=false`. No source stream was
  copied or modified by the audit.
- Admission remains **false**. The historical scope is annotated-cadence
  projection rather than a proven dense frame-online v2 stream;
  `primary_freeze_permitted=false`; exact private SAM release/training-stage
  binding is not verified; and no v2 frontend-stage receipt or shared v2 Val
  metric contract is attached.

This is a stronger completeness result for the local MASA candidate, not a
new OCD/MOT result and not permission to replace the selected COVTrack-native
route. Keep it quarantined until a separately registered v2 normalization and
legal frontend decision exist.
