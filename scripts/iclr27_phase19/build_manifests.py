"""Build legal Phase19 supported-known folds and ladder manifests.

The evaluator-only source CSV remains outside the trainer-facing manifest.  All
non-supported category values are replaced by -1 in the trainer audit table.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/iclr27_phase19/sources"
OUT = ROOT / "outputs/iclr27_phase19"


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def main() -> None:
    rows = list(csv.DictReader((SRC / "public_rows_corrected.csv").open(newline="")))
    supported = sorted(int(x) for x in json.loads((SRC / "supported_known_ids.json").read_text()))
    supported_set = set(supported)
    # Four fixed held sets.  They are selected from supported categories with
    # at least four videos and are never moved after training/public results.
    held_sets = [
        [805, 347, 229],
        [211, 235, 95],
        [382, 133, 579],
        [41, 429, 81],
    ]
    assert all(set(x) <= supported_set for x in held_sets)
    # Track and video summaries use only the evaluator-side builder here.  The
    # resulting trainer manifest contains masked semantic values.
    tracks: dict[str, list[int]] = defaultdict(list)
    track_video: dict[str, int] = {}
    track_cat: dict[str, int] = {}
    for i, r in enumerate(rows):
        key = f"v{int(r['video_id'])}:p{int(r['track_id'])}"
        tracks[key].append(i)
        track_video[key] = int(r["video_id"])
        cat = int(r["gt_category_id_common"])
        track_cat[key] = cat if cat in supported_set else -1
    for key in tracks:
        tracks[key].sort(key=lambda i: (int(rows[i]["event_rank"]), i))
    cat_videos: dict[int, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    for key, cat in track_cat.items():
        if cat >= 0:
            cat_videos[cat][track_video[key]].append(key)
    for d in cat_videos.values():
        for v in d:
            d[v].sort()

    fold_records = []
    for fold, held in enumerate(held_sets):
        held_set = set(held)
        train_categories = [c for c in supported if c not in held_set]
        # Per-category video split: first half train, final half validation;
        # with two videos this is exactly one video each.
        split = {}
        for c in supported:
            vids = sorted(cat_videos.get(c, {}))
            if not vids:
                continue
            cut = max(1, min(len(vids) - 1, int(round(len(vids) * .7)))) if len(vids) > 1 else 1
            split[str(c)] = {"train_videos": vids[:cut], "validation_videos": vids[cut:]}
        train_videos = sorted({v for c in train_categories if str(c) in split for v in split[str(c)]["train_videos"]})
        validation_videos = sorted({v for c in held if str(c) in split for v in split[str(c)]["validation_videos"]})
        fit_rows = [i for i, r in enumerate(rows) if int(r["video_id"]) in train_videos]
        # The model-facing rows contain only an opaque category sentinel.  The
        # evaluator category remains in the source CSV and is never copied into
        # this manifest.
        masked_rows = []
        for i in fit_rows:
            r = rows[i]
            cat = int(r["gt_category_id_common"])
            masked_rows.append({
                "row_index": i, "row_key": r["row_key"], "video_id": int(r["video_id"]),
                "tracklet_key": f"v{int(r['video_id'])}:p{int(r['track_id'])}",
                "frame_id": int(r["frame_id"]),
                "semantic_value_model": cat if cat in supported_set and cat not in held_set else -1,
                "assigned_model": int(r["assigned"] == "1"),
                "proposal_score": float(r["score"]),
                "causal_prefix_count": int(r["causal_prefix_count"]),
                "causal_box_stability_iou": float(r["causal_box_stability_iou"]),
            })
        # Track metadata is masked as well; the category is represented only by
        # a deterministic local index in the episode builder after loading.
        train_tracks = [k for k, c in track_cat.items() if track_video[k] in train_videos and c >= 0 and c not in held_set]
        held_tracks = [k for k, c in track_cat.items() if c in held_set and track_video[k] in validation_videos]
        fold_records.append({
            "fold": fold, "held_categories": held, "train_categories": train_categories,
            "train_videos": train_videos, "validation_videos": validation_videos,
            "fit_row_count": len(fit_rows), "fit_track_count": len(train_tracks),
            "held_track_count": len(held_tracks), "fit_row_indices_sha256": sha_bytes(",".join(map(str, fit_rows)).encode()),
            "train_track_keys_sha256": sha_bytes("\n".join(sorted(train_tracks)).encode()),
            "held_track_keys_sha256": sha_bytes("\n".join(sorted(held_tracks)).encode()),
            "category_video_split": split,
        })
        atomic(OUT / "manifests" / f"fold{fold}_masked_rows.json", {
            "fold": fold, "model_semantic_values_allowed": sorted(set(x["semantic_value_model"] for x in masked_rows)),
            "rows": masked_rows,
            "trainer_true_novel_values": [],
        })
    manifest = {
        "protocol": "trackocd_iclr27_phase19_strict_known_only_folds",
        "source_rows": len(rows), "supported_known_count": len(supported),
        "supported_known_ids_sha256": sha_bytes(",".join(map(str, supported)).encode()),
        "held_sets": held_sets, "folds": fold_records,
        "trainer_semantic_contract": {"allowed": supported + [-1], "true_novel_values_mapped_to": -1,
                                       "physical_id_as_semantic_value": False,
                                       "phase18_event_membership_read_by_trainer": False},
    }
    blob = json.dumps(manifest, sort_keys=True).encode()
    manifest["manifest_sha256"] = sha_bytes(blob)
    atomic(OUT / "manifests" / "fold_manifest.json", manifest)
    audit = {
        "rows": len(rows), "supported_ids": supported, "fold_count": 4,
        "all_true_novel_source_values_masked": True,
        "trainer_observed_semantic_values": sorted(set(x["semantic_value_model"] for x in masked_rows)),
        "true_novel_event_files_used_for_fold_construction": False,
        "video_disjoint_within_category": True,
        "ladders": {"L0": "reliable supported-known track prefixes", "L1": "high-quality proposal prefixes", "L2": "all causal proposal rows"},
    }
    atomic(OUT / "audit" / "legal_label_audit.json", audit)
    atomic(OUT / "manifests" / "symlink_inventory.json", {
        "sources": {p.name: os.readlink(p) for p in sorted(SRC.iterdir()) if p.is_symlink()},
        "large_assets_copied": False,
    })
    print(json.dumps({"complete": True, "manifest_sha256": manifest["manifest_sha256"], "folds": fold_records}, indent=2))


if __name__ == "__main__":
    main()
