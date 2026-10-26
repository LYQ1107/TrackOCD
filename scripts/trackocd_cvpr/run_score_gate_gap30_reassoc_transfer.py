#!/usr/bin/env python3
"""Run the fixed sparse-cadence causal reassociation transfer diagnostic.

The transform consumes only the existing anonymous COVTrack-native Train
streams.  It reuses the registered quality cut and changes only the causal
physical bridge horizon from the prior one-frame control to the preregistered
30-source-frame horizon.  It never reads GT, labels, vocabulary, pixels,
features, or Val/Test data, and writes to a new immutable namespace.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.trackocd_cvpr.causal_physical_reassociate import reassociate_video
from scripts.trackocd_cvpr.run_score_gate_reassoc_transfer import (
    _load_fixed_rows,
    _prepare_output_rows,
    _write_rows,
)

CONFIG = ROOT / "configs/trackocd_cvpr/covtrack_score_gate_gap30_reassoc_transfer_v1.json"
OUTPUT_ROOT = ROOT / "outputs/trackocd_cvpr/cvpr2/covtrack_score_gate_gap30_reassoc_transfer_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "trackocd.cvpr.covtrack-score-gate-gap30-reassociation-transfer.v1":
        raise ValueError("unexpected gap30 reassociation schema")
    if config.get("status") != "FROZEN_CATEGORY_FREE_SCORE_GATE_GAP30_REASSOCIATION_TRANSFER_NO_TRAINING":
        raise ValueError("gap30 reassociation scope is not frozen")
    mechanism = config.get("fixed_mechanism", {})
    if mechanism.get("score_threshold") != 0.005:
        raise ValueError("registered score threshold changed")
    if mechanism.get("reassociation_iou_threshold") != 0.5:
        raise ValueError("registered IoU threshold changed")
    if mechanism.get("reassociation_max_gap_frames") != 30:
        raise ValueError("registered sparse-cadence gap changed")
    if config.get("boundaries", {}).get("test_accessed") is not False:
        raise ValueError("Test boundary is not closed")
    if sha256_file(ROOT / config["source"]["reconciliation"]) != config["source"]["reconciliation_sha256"]:
        raise ValueError("source reconciliation changed")
    return config


def run(config_path: Path = CONFIG) -> dict[str, Any]:
    config = load_config(config_path)
    output_root = ROOT / "outputs/trackocd_cvpr/cvpr2/covtrack_score_gate_gap30_reassoc_transfer_v1"
    receipt = output_root / "run.json"
    if receipt.exists():
        raise RuntimeError(f"refusing to overwrite existing receipt: {receipt}")
    started = time.monotonic()
    roles = {
        role: {int(video_id) for video_id in spec["video_ids"]}
        for role, spec in config["fixed_roles"].items()
    }
    native, filtered, source_records = _load_fixed_rows(
        ROOT / config["source"]["stream_root"],
        roles,
        float(config["fixed_mechanism"]["score_threshold"]),
    )
    role_results: dict[str, Any] = {}
    for role in roles:
        native_rows = _prepare_output_rows(
            {video: list(rows) for video, rows in native[role].items()}
        )
        candidate_by_video: dict[int, list[dict[str, Any]]] = {}
        reassociation_by_video: dict[str, Any] = {}
        for video_id in sorted(roles[role]):
            rows, summary = reassociate_video(
                filtered[role].get(video_id, []),
                iou_threshold=float(config["fixed_mechanism"]["reassociation_iou_threshold"]),
                max_gap_frames=int(config["fixed_mechanism"]["reassociation_max_gap_frames"]),
            )
            candidate_by_video[video_id] = rows
            reassociation_by_video[str(video_id)] = summary
        candidate_rows = _prepare_output_rows(candidate_by_video)
        native_meta = _write_rows(
            ROOT / config["source"]["native_subset_outputs"][role], native_rows
        )
        candidate_meta = _write_rows(
            ROOT / config["source"]["candidate_outputs"][role], candidate_rows
        )
        role_results[role] = {
            "video_ids": sorted(roles[role]),
            "native_subset": native_meta,
            "candidate": candidate_meta,
            "native_tracks": len(native_rows),
            "native_observations": sum(len(row["frame_ids"]) for row in native_rows),
            "score_retained_tracks": sum(len(rows) for rows in filtered[role].values()),
            "score_retained_observations": sum(
                len(row["frame_ids"])
                for rows in filtered[role].values()
                for row in rows
            ),
            "reassociation_by_video": reassociation_by_video,
            "candidate_tracks": len(candidate_rows),
            "candidate_observations": sum(len(row["frame_ids"]) for row in candidate_rows),
        }

    result = {
        "schema_version": "trackocd.cvpr.covtrack-score-gate-gap30-reassociation-transfer-result.v1",
        "status": "COMPLETE_CATEGORY_FREE_SCORE_GATE_GAP30_REASSOCIATION_TRANSFER",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": {
            "path": str(config_path.resolve()),
            "sha256": sha256_file(config_path),
        },
        "source": {
            "root": str((ROOT / config["source"]["stream_root"]).resolve()),
            "reconciliation": str((ROOT / config["source"]["reconciliation"]).resolve()),
            "reconciliation_sha256": config["source"]["reconciliation_sha256"],
            "stream_count": len(source_records),
            "streams": source_records,
        },
        "roles": role_results,
        "mechanism": config["fixed_mechanism"],
        "boundaries": config["boundaries"],
        "resources": {
            "device": "CPU",
            "wall_seconds": time.monotonic() - started,
            "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        },
        "interpretation": (
            "Category-free sparse-cadence transfer evidence only; no official MOT/OCD metric, "
            "frontend qualification, model selection, predicted training, or Goal A/B/C acceptance."
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> int:
    result = run()
    print(json.dumps({
        "status": result["status"],
        "roles": {
            role: {
                "native_tracks": value["native_tracks"],
                "candidate_tracks": value["candidate_tracks"],
                "candidate_observations": value["candidate_observations"],
            }
            for role, value in result["roles"].items()
        },
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
