#!/usr/bin/env python3
"""Render the Phase58 raw-frame contract from the immutable inventory."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
inv=json.loads((ROOT/'outputs/iclr27_phase57/audit/supervision_inventory.json').read_text())
leak=json.loads((ROOT/'outputs/iclr27_phase57/audit/leakage_audit.json').read_text())
lines=['# Phase58 Raw-Frame Supervision Contract','',
'## Scope and boundary','',
'Only the public TRAIN TAO annotation file is used for loss/split metadata: `'+inv['source_annotation']+'`. The source hash is `'+inv['source_annotation_sha256']+'`. Pixels remain at the existing read-only frame root and are never copied into outputs. DEV+, Q1, public-new-model labels, held-event GT, future frames/tracks, category names/text and IDs as tensor features are excluded.', '',
'## Raw-frame availability','',
f"The TRAIN annotation contains {inv['images']} images, {inv['annotations']} boxes, {inv['videos']} videos, {inv['categories']} categories and {inv['tracks']} physical GT tracks. All referenced TRAIN frame paths were checked and missing count is {inv['missing_frame_count']}. The loader reads current/history images causally; GT boxes and category/track labels are detached loss metadata only.", '',
'## Four fixed disjoint folds','',
'Fit rows exclude both the fold held-category set and held-video set. Validation is the complementary set, so no training sample shares a held category or held video with fit. Exact held sets, counts and seeds are in `outputs/iclr27_phase57/manifests/fold_*.json`.', '',
'| fold | fit tracks | val tracks | fit videos | val videos | fit categories | cross-video positives | hard negatives | prefix tracks 1/2/4/8/16 |',
'|---:|---:|---:|---:|---:|---:|---:|---:|---|']
for f in inv['folds']:
 p='/'.join(str(f['prefix_track_coverage'][str(x)]) for x in (1,2,4,8,16))
 lines.append(f"| {f['fold']} | {f['fit_tracks']} | {f['validation_tracks']} | {f['fit_videos']} | {f['validation_videos']} | {f['fit_categories']} | {f['cross_video_positive_pairs']} | {f['hard_negative_pairs']} | {p} |")
lines += ['', '## Supervision tensors and labels', '',
'The model receives only current/history RGB crops or feature maps, candidate geometry, motion, age, quality, association confidence and causal masks. Loss-only metadata supplies objectness/box targets, same-track continuation/birth/termination, same-track temporal positives, cross-video multi-positive links, different-track hard negatives, prefix consistency, state persistence and Commit/Defer utility. Physical IDs and category IDs are used to construct these labels and folds, then removed before the forward pass.', '',
'Required causal graph:', '',
'`RGB frame prefix → class-agnostic objectness/box head → persistent physical query/association → lifecycle state → causal track encoder → 768-D raw-preserving semantic state → prior-only support bank → cross-video correspondence → semantic StateMemory → Commit/Defer/Reset`', '',
'## Leakage and protocol audit', '',
f"`leakage_audit.json` records future rows/tracks={leak['future_rows_or_tracks']}, held-event overlap={leak['held_event_overlap']}, sealed access={leak['devplus_q1_public_access']}, text inputs={leak['category_text_inputs']}, ID feature inputs={leak['semantic_or_physical_id_inputs']}, support/query overlap={leak['support_query_overlap']}, denominator drift={leak['denominator_drift']}, parent-assignment drift={leak['parent_assignment_drift']}. The original 76-event evaluator and row denominator remain a scoring-only comparator and are not used to train or choose checkpoints.", '',
'## Reproduction', '',
'`/home/lwr/anaconda3/envs/AVI/bin/python scripts/iclr27_phase57/build_frame_contract.py` reads only TRAIN annotations, writes manifests/JSON atomically, and creates `outputs/iclr27_phase57/completion/phase58_contract.done`. No feature-row NPZ is an input to the raw-frame model.']
(ROOT/'docs/iclr27_phase57/PHASE58_FRAME_SUPERVISION_CONTRACT.md').write_text('\n'.join(lines)+'\n')
