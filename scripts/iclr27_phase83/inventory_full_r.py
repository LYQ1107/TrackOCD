#!/usr/bin/env python3
"""Inventory the complete Phase30/75D R universe and physical-stream coverage.

This is an audit-only artifact.  It does not join held labels to model inputs
and does not alter the frozen Phase75D scorer.  The inventory intentionally
distinguishes the event-only native stream and the small historical TRAIN
stream from the full R universe; neither is silently treated as full
coverage.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase83"
CSV_PATH = ROOT / "outputs/iclr27_phase17r/csv/public_rows_corrected.csv"
NATIVE_EVENT = Path("/data2/usr_for_deadline/trackocd_phase75b/event_full_sequence_repair2/native_lineage.jsonl")
Q0_TRAIN_STREAM = ROOT / "outputs/iclr27_phase4t/train_stream/teta/tao_track.json"
EPISODES = ROOT / "outputs/iclr27_phase30/manifests"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def stream_inventory(path: Path) -> dict[str, Any]:
    rows = 0
    videos: set[int] = set()
    tracks: set[tuple[int, int]] = set()
    images: set[tuple[int, int]] = set()
    if not path.is_file():
        return {"path": str(path), "exists": False, "rows": 0, "videos": 0, "tracks": 0, "images": 0}
    with path.open(encoding="utf-8") as f:
        text = f.read(1)
        f.seek(0)
        if text == "[":
            payload = json.load(f)
            iterator = iter(payload)
        else:
            iterator = (json.loads(line) for line in f if line.strip())
        for d in iterator:
            rows += 1
            if not isinstance(d, dict) or d.get("video_id") is None:
                continue
            v = int(d["video_id"])
            videos.add(v)
            tid = d.get("physical_track_id", d.get("track_id"))
            if tid is not None:
                tracks.add((v, int(tid)))
            if d.get("image_id") is not None:
                images.add((v, int(d["image_id"])))
    return {"path": str(path.resolve()), "exists": True, "sha256": sha(path), "rows": rows, "videos": len(videos), "video_ids": sorted(videos), "tracks": len(tracks), "images": len(images)}


def main() -> None:
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    public_videos = sorted({int(r["video_id"]) for r in rows})
    public_tracks = sorted({(int(r["video_id"]), int(r["track_id"])) for r in rows})
    public_images = sorted({(int(r["video_id"]), int(r["image_id"])) for r in rows})
    queries: set[str] = set()
    fold_counts: dict[str, dict[str, int]] = {}
    for fold in range(4):
        p = EPISODES / f"episode_manifest_f{fold}.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        val = [r for r in d.get("records", []) if r.get("split") == "val"]
        q = {str(r["query_track_key"]) for r in val}
        queries.update(q)
        fold_counts[str(fold)] = {"validation_records": len(val), "validation_queries": len(q), "positive_records": sum(r.get("kind") == "multi_positive_cross_video" for r in val), "negative_records": sum(r.get("kind") == "null_no_match_hard_negative" for r in val)}
    native = stream_inventory(NATIVE_EVENT)
    q0_train = stream_inventory(Q0_TRAIN_STREAM)
    native_video_ids = set(native.get("video_ids", []))
    q0_train_video_ids = set(q0_train.get("video_ids", []))
    inventory = {
        "schema_version": "trackocd.phase83.r_full_coverage_inventory.v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "r_universe": {"rows": len(rows), "tracks": len(public_tracks), "images": len(public_images), "videos": len(public_videos), "video_ids": public_videos, "queries": len(queries), "query_keys": sorted(queries), "csv": str(CSV_PATH.resolve()), "csv_sha256": sha(CSV_PATH)},
        "fold_counts": fold_counts,
        "native_event_stream": native,
        "historical_q0_train_stream": q0_train,
        "coverage_against_r_universe": {
            "event_native_video_count": len(native_video_ids & set(public_videos)),
            "event_native_public_video_fraction": len(native_video_ids & set(public_videos)) / max(len(public_videos), 1),
            "historical_q0_train_public_video_count": len(q0_train_video_ids & set(public_videos)),
            "historical_q0_train_public_video_fraction": len(q0_train_video_ids & set(public_videos)) / max(len(public_videos), 1),
        },
        "contract": {"candidate_universe": "Phase30 validation tracks excluding same-video candidates", "prefixes": [1, 2, 4, 8, 16], "denominator_queries": len(queries), "raw_fallback_for_unmapped": "forbidden for A2 headline", "public_dev_q1_sealed_accessed": False, "future_rows_or_tracks": False, "ids_as_model_input": False},
        "status": "INVENTORY_COMPLETE_FULL_Q0_LINEAGE_REQUIRED",
    }
    atomic_json(OUT / "audit/r_full_coverage_inventory.json", inventory)
    atomic_json(OUT / "status.json", {"phase": "Phase83", "status": inventory["status"], "next_action": "generate full Q0 physical lineage for all R-universe videos, then corrected DINO coverage and A2 temporal replay", "inventory": str((OUT / "audit/r_full_coverage_inventory.json").resolve()), "public_dev_q1_sealed_accessed": False, "resource_event": "read_only_cpu"})
    print(json.dumps({"status": inventory["status"], "rows": len(rows), "tracks": len(public_tracks), "queries": len(queries), "native_event_public_videos": len(native_video_ids & set(public_videos)), "historical_q0_train_public_videos": len(q0_train_video_ids & set(public_videos))}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
