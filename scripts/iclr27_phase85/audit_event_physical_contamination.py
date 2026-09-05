#!/usr/bin/env python3
"""Audit physical-root/category contamination on the fixed event videos."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OBS = Path("/data2/usr_for_deadline/trackocd_phase75b/observability_repair2/event_observability.jsonl")
PUBLIC = ROOT / "data/iclr27_phase19r/sources/public_rows_corrected.csv"
Q0 = Path("/data2/usr_for_deadline/trackocd_phase83/a2_full/native_lineage.jsonl")
TEMPORAL = Path("/data2/usr_for_deadline/trackocd_phase85/project_outputs/physical/temporal_mean_full/full_temporal_lineage.jsonl")
SELECTIVE = Path("/data2/usr_for_deadline/trackocd_phase85/project_outputs/physical/selective_formal_r1/full_temporal_lineage.jsonl")
OUT = ROOT / "outputs/iclr27_phase85"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def root_of(row: dict[str, Any]) -> tuple[int, int]:
    return (int(row.get("video_id", -1)), int(row.get("phase85_canonical_physical_track_id", row.get("physical_track_id", -1))))


def as_int(value: Any, default: int = -1) -> int:
    try:
        return int(value) if value not in (None, "", "None") else default
    except (TypeError, ValueError):
        return default


def load_event_videos() -> set[int]:
    vids: set[int] = set()
    for line in OBS.open(encoding="utf-8"):
        if line.strip():
            x = json.loads(line); vids.update({int(x.get("source_video", -1)), int(x.get("target_video", -1))})
    return {v for v in vids if v >= 0}


def load_lineage(path: Path, event_videos: set[int], key_set: set[tuple[str, int, int]]) -> tuple[dict[tuple[str, int, int], tuple[int, int]], Counter]:
    roots: dict[tuple[str, int, int], tuple[int, int]] = {}
    stats = Counter()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line); video = int(row.get("video_id", -1))
            if video not in event_videos: continue
            key = (str(row.get("file_path", "")), as_int(row.get("frame_id")), as_int(row.get("proposal_local_id")))
            if key in key_set:
                roots[key] = root_of(row); stats["mapped"] += 1
            stats["rows_event_videos"] += 1
    return roots, stats


def main() -> None:
    event_videos = load_event_videos()
    public = list(csv.DictReader(PUBLIC.open(newline="", encoding="utf-8")))
    key_categories: dict[tuple[str, int, int], set[int]] = defaultdict(set)
    public_event = []
    for row in public:
        video = int(row["video_id"])
        if video not in event_videos: continue
        key = (str(row.get("image_path", "")), as_int(row.get("frame_id")), as_int(row.get("proposal_local_id")))
        try: cat = int(float(row.get("gt_category_id_common", -1) or -1))
        except Exception: cat = -1
        if cat >= 0: key_categories[key].add(cat)
        public_event.append((key, video, cat, str(row.get("row_key"))))
    key_set = set(key for key, _, _, _ in public_event)
    lineage = {}
    stats = {}
    for name, path in (("q0", Q0), ("temporal_mean", TEMPORAL), ("selective", SELECTIVE)):
        lineage[name], stats[name] = load_lineage(path, event_videos, key_set)
    result_rows = []
    for key, video, cat, row_key in public_event:
        result_rows.append({"row_key": row_key, "video_id": video, "image_path": key[0], "frame_id": key[1], "proposal_local_id": key[2], "category": cat, "q0_root": lineage["q0"].get(key), "temporal_root": lineage["temporal_mean"].get(key), "selective_root": lineage["selective"].get(key)})
    root_sets: dict[str, dict[tuple[int, int], set[int]]] = {name: defaultdict(set) for name in lineage}
    for row in result_rows:
        cat = row["category"]
        if cat < 0: continue
        for name in lineage:
            field = "temporal_root" if name == "temporal_mean" else f"{name}_root"
            root = row.get(field)
            if root is not None: root_sets[name][tuple(root)].add(cat)
    summary = {}
    for name, roots in root_sets.items():
        counts = Counter(len(cats) for cats in roots.values())
        cross = sum(1 for cats in roots.values() if len(cats) > 1)
        field = "temporal_root" if name == "temporal_mean" else f"{name}_root"
        summary[name] = {"roots_with_category": len(roots), "multi_category_roots": cross, "multi_category_fraction": cross / max(len(roots), 1), "category_cardinality": dict(sorted((str(k), v) for k, v in counts.items())), "event_rows_mapped": sum(r.get(field) is not None for r in result_rows)}
    changes = {}
    for a, b in (("q0", "temporal_mean"), ("q0", "selective"), ("temporal_mean", "selective")):
        field_a = "temporal_root" if a == "temporal_mean" else f"{a}_root"
        field_b = "temporal_root" if b == "temporal_mean" else f"{b}_root"
        changes[f"{a}_to_{b}"] = sum(r.get(field_a) != r.get(field_b) for r in result_rows)
    result = {"schema_version": "trackocd.phase85.event_physical_contamination.v1", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "event_video_count": len(event_videos), "event_videos": sorted(event_videos), "public_event_rows": len(public_event), "lineages": {"q0": {"path": str(Q0.resolve()), "sha256": sha(Q0)}, "temporal_mean": {"path": str(TEMPORAL.resolve()), "sha256": sha(TEMPORAL)}, "selective": {"path": str(SELECTIVE.resolve()), "sha256": sha(SELECTIVE)}}, "lineage_scan_stats": {name: dict(value) for name, value in stats.items()}, "summary": summary, "root_changes": changes, "rows": result_rows, "labels_posthoc_only": True, "public_dev_q1_sealed_accessed": False, "future_rows_or_tracks": False, "ids_as_model_input": False}
    atomic(OUT / "audit/event_physical_contamination.json", result)
    atomic(OUT / "completion/event_physical_contamination.done", {"status": "DONE", "audit": str((OUT / "audit/event_physical_contamination.json").resolve()), "sha256": sha(OUT / "audit/event_physical_contamination.json")})
    print(json.dumps({"event_video_count": len(event_videos), "public_event_rows": len(public_event), "summary": summary, "root_changes": changes}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
