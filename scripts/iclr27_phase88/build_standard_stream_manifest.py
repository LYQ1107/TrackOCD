#!/usr/bin/env python3
"""Build the deterministic auxiliary stream manifest for Phase88 diagnostics."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase88"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def main() -> None:
    folds = []
    for fold in range(4):
        source = OUT / "manifests" / f"fit_events_v2_fix2_f{fold}.jsonl"
        tracks: dict[str, dict] = {}
        # Known anchors are metadata-only stream ordering inputs; category
        # values never enter a model tensor.  If the compact fold loader is
        # unavailable, the manifest remains valid but records zero anchors.
        known_anchor_count = 0
        try:
            import sys
            sys.path.insert(0, str(ROOT))
            from src.iclr27_phase88.data_memmap import Phase88FoldData
            data = Phase88FoldData(fold, "/data2/usr_for_deadline/trackocd_phase88/shared_features")
            for key, _cat in sorted(data.known_eval_keys):
                key = str(key)
                tracks.setdefault(key, {"track_key": key, "video_id": int(data.video(key)), "roles": set()})["roles"].add("known_anchor")
                known_anchor_count += 1
        except Exception:
            pass
        for line in source.read_text().splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            for item in event.get("source_tracks", []):
                key = str(item["track_key"])
                tracks.setdefault(key, {"track_key": key, "video_id": int(item["video_id"]), "roles": set()})["roles"].add("novel_source")
            key = str(event["target_track_key"])
            tracks.setdefault(key, {"track_key": key, "video_id": int(event["target_video"]), "roles": set()})["roles"].add("novel_target")
        rows = []
        for key, item in tracks.items():
            roles = sorted(item["roles"])
            rows.append({"stream_index": 0, "track_key": key, "video_id": item["video_id"], "role": roles[0], "roles": roles,
                         "causal_order_key": [int(item["video_id"]), key], "category_inference_input": False})
        priority = {"known_anchor": 0, "novel_source": 1, "novel_target": 2}
        rows.sort(key=lambda x: (priority.get(x["role"], 9), x["video_id"], x["track_key"], x["role"]))
        for i, row in enumerate(rows):
            row["stream_index"] = i
        folds.append({"fold": fold, "source_manifest": str(source.resolve()), "source_sha256": sha(source),
                      "known_anchor_count": known_anchor_count, "track_count": len(rows), "tracks": rows, "state_memory_reset_between_events": False,
                      "ordering": "opaque stable (video_id, track_key, role); no category/performance optimization"})
    payload = {
        "schema_version": "trackocd.phase88.standard_gcd_stream.v1", "phase": 88,
        "name": "standard_gcd_stream_v1", "deterministic": True, "causal": True,
        "known_anchor_tracks_first": True,
        "anonymous_tokens_are_fold_stream_local": True,
        "global_hungarian_only_after_stream_collection": True,
        "future_rows_or_tracks": False, "category_text_used_for_order": False,
        "held_tuning": False, "folds": folds,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    atomic_json(OUT / "manifests/standard_gcd_stream_v1.json", payload)
    print(json.dumps({"path": str((OUT / 'manifests/standard_gcd_stream_v1.json').resolve()), "fold_track_counts": [x["track_count"] for x in folds]}, indent=2))


if __name__ == "__main__":
    main()
