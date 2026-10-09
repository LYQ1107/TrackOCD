#!/usr/bin/env python3
"""Preregister separate M1 all-Val frozen physical audit, not M9 or DINO cache."""
from __future__ import annotations
import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import atomic_json, sha256_file
from src.trackocd_core.physical_qualification import metadata_universe, val_image, json_digest


def main():
    started = time.monotonic()
    private = ROOT / "outputs/trackocd_core/audit/masa_physical_qualification_plan.json"
    public = ROOT / "configs/trackocd_core/masa_physical_qualification.json"
    if private.exists() or public.exists():
        raise ValueError("Preserve original all-universe plan; no resampling")
    annotation = Path("/data3/liuyeqiang/TAO-Amodal/annotations/validation.json")
    expected = "0414885ee2702c2d3176cf6184e7811a7bd1c1347a157fef57a91020976776ee"
    if sha256_file(annotation) != expected:
        raise ValueError("Known Val metadata source changed")
    videos = metadata_universe(json.loads(annotation.read_text()))
    if len(videos) != 988 or sum(len(v["images"]) for v in videos) != 36375:
        raise ValueError("Not the entire canonical Val universe")
    frames = Path("/data3/liuyeqiang/TAO-Amodal/frames")
    logical = 0
    for video in videos:
        for image in video["images"]:
            path = val_image(frames, image["image_path"])
            image.update(image_sha256=sha256_file(path), image_bytes=path.stat().st_size)
            logical += image["image_bytes"]
    plan = {"schema_version": "trackocd.core.masa_physical_all_val_plan.v1", "videos": videos,
            "selection": "Entire video/image metadata universe; sorted video ID and strict chronological frame_index, no label/annotation selection"}
    native = json.loads((ROOT / "outputs/trackocd_core/audit/masa_native_smoke.json").read_text())
    diag = json.loads((ROOT / "configs/trackocd_core/masa_physical_diagnostic.json").read_text())
    diag_result = json.loads((ROOT / "outputs/trackocd_core/audit/masa_physical_diagnostic_result.json").read_text())
    atomic_json(private, plan)
    config = {"schema_version": "trackocd.core.masa_physical_qualification.v1", "status": "PREREGISTERED_NOT_EXECUTED",
              "scope": "M1 full-Val frozen physical candidate audit on annotated cadence; not M9 OCD evaluation, semantic training or full DINO cache",
              "authorization_basis": "Expanded FINAL_GOAL M1 requires Val physical audit; completed Train8 engineering and separate fixed Val64 diagnostic precede this stage. M8 small-Train semantic-first boundary remains unchanged.",
              "selection": plan["selection"], "private_plan_sha256": sha256_file(private),
              "videos": len(videos), "images": 36375, "image_counts": {str(v["video_id"]): len(v["images"]) for v in videos},
              "annotation_sha256": expected, "roles_sha256": sha256_file(ROOT / "configs/trackocd_core/roles.json"),
              "canonical_adapter_sha256": diag_result["canonical_adapter_sha256"],
              "canonical_hota_sha256": diag_result["canonical_hota_sha256"],
              "native_checkpoint_sha256": native["frozen_model"]["checkpoint_sha256"],
              "native_model_config_sha256": json_digest(native["frozen_model"]["native_model_config"]),
              "candidate_model_unchanged_from_completed_smoke_and_val4": True,
              "roi_cap": 50, "roi_score_threshold": .02, "new_weight_or_training_or_threshold_search": False,
              "reuse": "Verify config/entire-video image plan/hash/array integrity of completed video markers before skipping. Val4 clips lack full-video state and are not full-video shards; preserve them. Interrupted incomplete video must restart its causal state, with old attempt retained and counted; never re-extract a valid completed video.",
              "cadence_boundary": diag["sampling_boundary"].replace("selected", "entire-video registered"),
              "provenance": {"grade": "OFFICIAL_PUBLICATION_AND_PINNED_RELEASE_GENERIC_IMAGE_ROUTE_WITH_DISCLOSED_PRIVATE_STAGE_GAP",
                             "paper": "https://arxiv.org/html/2406.04221v1", "paper_sections": ["4.1", "J.1", "J.2"],
                             "sam_foundation_author_statement": "https://github.com/facebookresearch/segment-anything/issues/53",
                             "reported_training": "Main models: SA-1B-500K, frozen official SAM foundation; SAM track-head-only final6-epoch phase disclosed in J.2; other BDD/COCO and TETer domain-adaptation experiments are separate, not automatically release lineage",
                             "pinned_route_receipt": "outputs/trackocd_core/audit/masa_training_route.json",
                             "exact_private_sam_release_training_stage_binding_verified": False,
                             "forbidden_supervision_proven": False,
                             "runtime": "One-class native RPN/ROI, no Detic, vocabulary, semantic checkpoint metadata fallback, text, GT, public detections or offline filtering"},
              "evaluation": {"hota": "Same pinned TAO_OW SUBSET=all/MAX_DETECTIONS=300/unchanged partial-annotation preprocessing; per-video canonical evaluation and count-weighted combine_sequences",
                             "coverage": "Full inherited Known/Novel GT track denominators, fixed class-free per-video temporal-IoU>=0.5 Hungarian; missing predictions not removed",
                             "purity": "Per-frame maximum-cardinality then IoU>=0.5 geometry join; unknown unmatched observations retained, category/identity mixing only posthoc",
                             "lengths": "All annotated-cadence observations; report as annotated projection, not dense/full-frame video lifetime",
                             "comparison": "Existing frozen PANDAS BT-FG-0 full-Val reference/projection only; different association cadence/history, not a matched-input tracker ablation",
                             "labels_for_model_input_or_tuning": False, "automatic_primary_freeze": False,
                             "role_threshold_sweep_or_semantic_model_selection": False},
              "asset_inventory": {"existing_required_image_bytes": logical, "new_image_copy_bytes": 0,
                                  "existing_checkpoint_bytes": 558882875, "new_weight_bytes": 0,
                                  "prediction_uncompressed_payload_upper_bytes": 36375 * 50 * (2 * (4 * 4 + 4) + 8) + 36375 * 32,
                                  "conservative_prediction_and_cpu_eval_output_ceiling_bytes": 1073741824},
              "limits": {"max_images": 36375, "max_videos": 988, "max_gpu_workers": 4, "max_gpu_bytes_per_worker": 8589934592,
                         "max_host_rss_bytes_per_worker": 4294967296, "max_output_allocated_bytes": 1073741824,
                         "max_candidate_allocated_bytes": 8589934592, "minimum_ram_headroom_fraction": .25,
                         "minimum_disk_headroom_bytes": 4294967296, "overall_soft_bytes": 16106127360,
                         "overall_hard_bytes": 32212254720, "arbitrary_wall_time_cutoff": False},
              "stop_rules": ["Resource headroom, identity, finite/valid current input/output or model immutability violation: stop only newly owned workers and preserve attempts",
                             "No skipping failed/empty/low-coverage videos, no shrinking denominator, no resampling or candidate configuration change",
                             "All complete video markers verified and full stream sealed before first independent GT scoring",
                             "Quality/provenance must be reported honestly; no automatic qualification, M9 claim or Novel-GT-driven semantic/physical tuning"]}
    atomic_json(public, config)
    print(json.dumps({"videos": len(videos), "images": config["images"], "private_plan_sha256": config["private_plan_sha256"],
                      "existing_image_bytes": logical, "wall_seconds": time.monotonic() - started,
                      "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}))


if __name__ == "__main__":
    main()
