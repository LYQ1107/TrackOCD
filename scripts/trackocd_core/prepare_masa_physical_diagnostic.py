#!/usr/bin/env python3
"""Seal label-blind metadata prefixes before a separate bounded M1 diagnostic."""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_core.physical_diagnostic import metadata_prefixes
from src.trackocd_v2.io import atomic_json, sha256_file


def main() -> None:
    private = ROOT / "outputs/trackocd_core/audit/masa_physical_diagnostic_plan.json"
    public = ROOT / "configs/trackocd_core/masa_physical_diagnostic.json"
    if private.exists() or public.exists():
        raise ValueError("Preserve sealed diagnostic plan; do not resample")
    annotation = Path("/data3/liuyeqiang/TAO-Amodal/annotations/validation.json")
    expected_sha = "0414885ee2702c2d3176cf6184e7811a7bd1c1347a157fef57a91020976776ee"
    if sha256_file(annotation) != expected_sha:
        raise ValueError("Known Val source changed")
    selected = metadata_prefixes(json.loads(annotation.read_text()))
    frames = Path("/data3/liuyeqiang/TAO-Amodal/frames")
    for video in selected:
        for image in video["images"]:
            relative = Path(image["image_path"])
            path = frames / relative
            if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "val" or not path.resolve().is_relative_to((frames / "val").resolve()):
                raise ValueError("Unexpected Val image path")
            image["image_sha256"] = sha256_file(path)
    plan = {"schema_version": "trackocd.core.masa_physical_private_plan.v1", "videos": selected,
            "selection": "First four ascending Val video IDs; first at most16 image metadata rows by frame_index/image_id, no category/annotation inspection or substitution"}
    atomic_json(private, plan)
    native = json.loads((ROOT / "outputs/trackocd_core/audit/masa_native_smoke.json").read_text())
    config = {"schema_version": "trackocd.core.masa_physical_diagnostic.v1", "status": "PREREGISTERED_NOT_EXECUTED",
              "scope": "M1 bounded frozen physical quality diagnostic, not M9, full-Val result or primary freeze",
              "selection": plan["selection"], "private_plan_sha256": sha256_file(private),
              "video_ids": [v["video_id"] for v in selected], "images_per_video": [len(v["images"]) for v in selected],
              "frame_indices": [[i["frame_index"] for i in v["images"]] for v in selected],
              "annotation_sha256": expected_sha, "native_checkpoint_sha256": native["frozen_model"]["checkpoint_sha256"],
              "native_model_config_sha256": hashlib.sha256(json.dumps(native["frozen_model"]["native_model_config"],sort_keys=True,separators=(",", ":")).encode()).hexdigest(),
              "candidate_model_unchanged_from_completed_smoke": True,
              "new_weight_or_training_or_threshold_search": False,
              "provenance_boundary": "Published generic-image route supported; exact released SAM-B training stage config/data ledger missing. Diagnostic is not supervision certification.",
              "sampling_boundary": "Native state resets at first selected annotated frame, annotated-only ordinal cadence. Frozen PANDAS projection keeps dense-frame association/prehistory; shared scoring frames, not a matched-input association ablation.",
              "candidate": "released MASA-SAM-B native anonymous proposal wrapper, not official public-Detic benchmark",
              "comparison": "Existing frozen PANDAS BT-FG-0 boxes/IDs projected without rerunning/tuning its detector/tracker",
              "evaluation": {"hota": "Existing pinned TAO_OW adapter SUBSET=all/MAX_DETECTIONS=300, unchanged preprocessing; identical clipped GT for both streams",
                             "coverage": "Fixed class-free temporal-IoU>=0.5 Hungarian on entire selected-clip Known/Novel GT universe; all predictions retained",
                             "purity": "Per-frame geometric match first; categories/identity mixing aggregate only after prediction sealing. Unknown unmatched rows not declared background/pure.",
                             "lengths": "Native annotated-cadence lengths and PANDAS annotated projection lengths only; not full-video lifetime lengths",
                             "categories_for_input_or_tuning": False, "matched_only_headline": False, "full_val_or_ocd_metrics_claim": False},
              "limits": {"max_images":64, "max_videos":4, "max_gpu_workers":1,"max_seconds":600,"max_host_rss_bytes":4294967296,
                         "max_gpu_bytes":8589934592,"max_output_bytes":10485760,"max_candidate_allocated_bytes":8589934592,
                         "minimum_ram_headroom_fraction":.25,"overall_soft_bytes":16106127360,"overall_hard_bytes":32212254720},
              "stop_rules": ["Fresh resource/identity/strict-weight/input-finiteness checks must pass", "No replacing empty/missing/low-coverage videos or extending to get more Novel targets", "Preserve empty/failed output; no physical tuning, native-smoke rerun, semantic training, Test or full-Val/cache job", "No primary freeze from four clipped videos, regardless of scores"]}
    atomic_json(public, config)
    print(json.dumps({k:config[k] for k in ("scope","video_ids","images_per_video","private_plan_sha256")}))


if __name__ == "__main__":
    main()
