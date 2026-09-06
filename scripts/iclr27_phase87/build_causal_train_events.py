#!/usr/bin/env python3
"""Build monotonic TRAIN-only causal events for the Phase87 controller."""
from __future__ import annotations

import csv
import hashlib
import json
import random
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "iclr27_phase87"
CSV_PATH = ROOT / "outputs" / "iclr27_phase17r" / "csv" / "public_rows_corrected.csv"
FOLD_MANIFEST = ROOT / "outputs" / "iclr27_phase19r" / "manifests" / "fold_manifest.json"
EVENT_OBS = Path("/data2/usr_for_deadline/trackocd_phase75b/observability_repair2/event_observability.jsonl")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atom(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp = Path(handle.name)
    temp.replace(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        temp = Path(handle.name)
    temp.replace(path)


def first_reliable(item: dict) -> int:
    for position, row in enumerate(item["row_indices"][:16], start=1):
        try:
            if int(row.get("assigned", 0)) == 1 and float(row.get("row_iou", 0.0)) >= 0.5:
                return position
        except (TypeError, ValueError):
            continue
    return min(16, int(item["length"]))


def main() -> None:
    rows = list(csv.DictReader(CSV_PATH.open(newline="")))
    tracks: dict[str, dict] = {}
    for row in rows:
        if row.get("gt_category_id_common") in {"", "-1", "None", None}:
            continue
        key = f"v{int(row['video_id'])}:p{int(row['track_id'])}"
        item = tracks.setdefault(key, {"key": key, "video": int(row["video_id"]), "category": int(row["gt_category_id_common"]), "length": 0, "row_indices": []})
        item["length"] += 1
        item["row_indices"].append(row)
    excluded: set[int] = set()
    for line in EVENT_OBS.open():
        if line.strip():
            obj = json.loads(line)
            excluded.update((int(obj.get("source_video", -1)), int(obj.get("target_video", -1))))
    valid = {key: item for key, item in tracks.items() if item["video"] not in excluded}
    by_category: dict[int, list[str]] = defaultdict(list)
    for key, item in valid.items():
        by_category[item["category"]].append(key)
    for values in by_category.values():
        values.sort(key=lambda key: (valid[key]["video"], key))
    folds = json.loads(FOLD_MANIFEST.read_text())["folds"]
    rng = random.Random(87001)
    all_counts = {}
    for fold_info in folds:
        fold = int(fold_info["fold"])
        held_categories = {int(x) for x in fold_info["held_categories"]}
        fit_videos = {int(x) for x in fold_info["fit_videos"]}
        val_videos = {int(x) for x in fold_info["validation_videos"]}
        fit_keys = [key for key, item in valid.items() if item["video"] in fit_videos and item["category"] not in held_categories]
        val_keys = [key for key, item in valid.items() if item["video"] in val_videos or item["category"] in held_categories]
        fit_set = set(fit_keys)
        val_set = set(val_keys)

        def make_events(target_keys: list[str], source_keys: set[str], split: str, limit: int) -> list[dict]:
            positives, negatives = [], []
            source_by_cat: dict[int, list[str]] = defaultdict(list)
            for key in source_keys:
                source_by_cat[valid[key]["category"]].append(key)
            for key in source_by_cat:
                source_by_cat[key].sort(key=lambda x: (valid[x]["video"], x))
            for target in sorted(target_keys):
                item = valid[target]
                prior_same = [source for source in source_by_cat[item["category"]] if valid[source]["video"] < item["video"]]
                prior_diff = [source for source in source_keys if valid[source]["video"] < item["video"] and valid[source]["category"] != item["category"]]
                if prior_same:
                    source = prior_same[-1]
                    positives.append({"event_id": f"f{fold}-{split}-pos-{len(positives):05d}", "fold": fold, "split": split, "polarity": "positive", "source_track_keys": [source], "target_track_key": target, "source_video": valid[source]["video"], "target_video": item["video"], "target_category_for_loss_only": item["category"], "reliable_prefix_for_loss_only": first_reliable(item), "source_before_target": True, "metadata_only": ["category", "track_key", "video_id", "reliable_prefix"]})
                if prior_diff:
                    # Deterministic hard-negative proxy: nearest video/category ordering,
                    # with no labels or IDs entering model tensors.
                    prior_diff = sorted(prior_diff, key=lambda source: (abs(valid[source]["video"] - item["video"]), valid[source]["category"], source))
                    source = prior_diff[0]
                    negatives.append({"event_id": f"f{fold}-{split}-neg-{len(negatives):05d}", "fold": fold, "split": split, "polarity": "negative", "source_track_keys": [source], "target_track_key": target, "source_video": valid[source]["video"], "target_video": item["video"], "target_category_for_loss_only": item["category"], "reliable_prefix_for_loss_only": first_reliable(item), "source_before_target": True, "metadata_only": ["category", "track_key", "video_id", "reliable_prefix"]})
            rng.shuffle(positives)
            rng.shuffle(negatives)
            return (positives + negatives)[:limit]

        train = make_events(fit_keys, fit_set, "fit", 12000)
        validation = make_events(val_keys, fit_set | val_set, "val", 2500)
        write_jsonl(OUT / "manifests" / f"train_events_f{fold}.jsonl", train)
        write_jsonl(OUT / "manifests" / f"val_events_f{fold}.jsonl", validation)
        all_counts[str(fold)] = {"fit_tracks": len(fit_keys), "val_tracks": len(val_keys), "train_events": len(train), "val_events": len(validation), "train_positive": sum(e["polarity"] == "positive" for e in train), "train_negative": sum(e["polarity"] == "negative" for e in train), "val_positive": sum(e["polarity"] == "positive" for e in validation), "val_negative": sum(e["polarity"] == "negative" for e in validation)}
    manifest = {"schema_version": "trackocd.phase87.causal_event_manifest.v1", "phase": 87, "protocol": "strict monotonic source-before-target TRAIN-only events", "csv_sha256": sha(CSV_PATH), "fold_manifest_sha256": sha(FOLD_MANIFEST), "event_observability_path": str(EVENT_OBS), "excluded_event_video_count": len(excluded), "excluded_event_videos": sorted(excluded), "counts": all_counts, "model_input_fields": ["causal_raw_sequence", "causal_geometry_sequence", "quality_sequence", "support_features", "causal_state"], "loss_only_fields": ["polarity", "target_category_for_loss_only", "reliable_prefix_for_loss_only", "source_track_keys", "video_id"], "future_rows_or_tracks": False, "ids_or_text_as_model_input": False, "held_event_overlap": False}
    atom(OUT / "manifests" / "causal_event_manifest.json", manifest)
    atom(OUT / "audit" / "causal_event_build.json", manifest)
    atom(OUT / "completion" / "causal_events.done", {"status": "DONE", "manifest": str((OUT / "manifests" / "causal_event_manifest.json").resolve())})
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
