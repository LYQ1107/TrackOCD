#!/usr/bin/env python3
"""Audit lawful cross-video Known support before an RCM training repair.

This is a read-only census over the already qualified policy Train cache and
its approved external loss targets.  It does not construct the RCM model,
decode pixels, run inference, optimize, or expose category identifiers.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.trackocd_cvpr.policy_training_data import load_completed_policy_videos

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/trackocd_cvpr/rcm_support_availability_audit_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/trackocd_cvpr/cvpr5/rcm_support_availability_audit_20261026.json"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "trackocd.cvpr.rcm-support-availability-audit.v1":
        raise ValueError("unexpected RCM support-availability configuration")
    if config.get("status") != "FROZEN_READ_ONLY_RCM_SUPPORT_AVAILABILITY_AUDIT":
        raise ValueError("support-availability audit is not frozen")
    boundaries = config.get("boundaries", {})
    if boundaries.get("read_only") is not True or any(
        boundaries.get(key) is not False
        for key in (
            "model_constructed", "inference_started", "optimizer_started", "new_cache_started",
            "gt_category_fields_read", "novel_vocabulary_read", "val_labels_read", "test_accessed",
            "model_selection", "semantic_ids_written_to_receipt",
        )
    ):
        raise ValueError("support-availability audit boundary is open")
    return config


def audit(config_path: Path, output_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    rcm_config_path = ROOT / config["joint_rcm_config"]
    coverage_path = ROOT / config["coverage_receipt"]
    if _sha256(rcm_config_path) != config["joint_rcm_config_sha256"]:
        raise ValueError("joint RCM config hash changed")
    if _sha256(coverage_path) != config["coverage_receipt_sha256"]:
        raise ValueError("RCM coverage receipt hash changed")
    rcm_config = json.loads(rcm_config_path.read_text(encoding="utf-8"))
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    if coverage.get("status") != "FAIL_RCM_PILOT_M_SUPPORT_COVERAGE_ZERO":
        raise ValueError("support census must bind the preserved zero-coverage receipt")

    videos, projection, proof = load_completed_policy_videos(ROOT, rcm_config, lambda: None)
    ordered_videos = [int(video) for video, _arrays in videos]
    known_by_video: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    pseudo_by_video: dict[int, int] = defaultdict(int)
    masked_by_video: dict[int, int] = defaultdict(int)
    physical_track_keys = [
        (int(video), int(track_id))
        for video, arrays in videos
        for track_id in arrays["track_id"]
    ]
    for video, track in physical_track_keys:
        target = projection.targets.get((video, track), ("no_retained_semantic_truth", None))
        if target[0] == "Known":
            known_by_video[video][int(target[1])] += 1
        elif target[0] == "policy_pseudo":
            pseudo_by_video[video] += 1
        else:
            masked_by_video[video] += 1

    pairs: list[dict[str, int]] = []
    for left_ordinal, left_video in enumerate(ordered_videos):
        for right_ordinal in range(left_ordinal + 1, len(ordered_videos)):
            right_video = ordered_videos[right_ordinal]
            shared = set(known_by_video[left_video]) & set(known_by_video[right_video])
            if shared:
                pairs.append({
                    "left_ordinal": left_ordinal,
                    "right_ordinal": right_ordinal,
                    "shared_known_binding_count": len(shared),
                    "shared_known_track_overlap_count": sum(
                        min(known_by_video[left_video][binding], known_by_video[right_video][binding])
                        for binding in shared
                    ),
                })
    first = pairs[0] if pairs else None
    result = {
        "schema_version": "trackocd.cvpr.rcm-support-availability-result.v1",
        "status": "PASS_LAWFUL_CROSS_VIDEO_KNOWN_SUPPORT_AVAILABLE" if pairs else "FAIL_NO_LAWFUL_CROSS_VIDEO_KNOWN_SUPPORT",
        "generated_utc": _now(),
        "config": {"path": str(config_path.resolve()), "sha256": _sha256(config_path)},
        "source": {
            "joint_rcm_config_sha256": _sha256(rcm_config_path),
            "coverage_receipt_sha256": _sha256(coverage_path),
            "policy_video_count": len(ordered_videos),
        },
        "support": {
            "known_target_tracks": sum(sum(known_by_video[v].values()) for v in ordered_videos),
            "policy_pseudo_target_tracks": sum(pseudo_by_video.values()),
            "masked_target_tracks": sum(masked_by_video.values()),
            "physical_track_count": len(physical_track_keys),
            "videos_with_known_targets": sum(bool(known_by_video[v]) for v in ordered_videos),
            "known_tracks_by_frozen_policy_order": [sum(known_by_video[v].values()) for v in ordered_videos],
            "cross_video_known_support_pair_count": len(pairs),
            "first_support_pair_by_frozen_ordinals": first,
        },
        "preserved_rcm_coverage": {
            "m_loss_steps": coverage["coverage"]["m_loss_steps"],
            "positive_other_video_groups": coverage["coverage"]["positive_other_video_groups"],
            "pilot_status": coverage["source"]["pilot_status"],
        },
        "boundaries": config["boundaries"],
        "interpretation": "Lawful external Known support exists in the registered policy fold, while the preserved RCM pilot produced zero prediction-owned cross-video support groups. This narrows the repair target to anonymous lifecycle grouping/coverage; it is not M-loss evidence, fold selection, or scientific acceptance.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = audit(args.config.resolve(), args.output.resolve())
    print(json.dumps({
        "status": result["status"],
        "output": str(args.output.resolve()),
        "cross_video_known_support_pair_count": result["support"]["cross_video_known_support_pair_count"],
        "preserved_m_loss_steps": result["preserved_rcm_coverage"]["m_loss_steps"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
