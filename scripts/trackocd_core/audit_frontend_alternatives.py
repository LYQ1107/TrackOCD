#!/usr/bin/env python3
"""Bounded pinned-source preflight; no upstream import, weights or dataset access."""
from __future__ import annotations

import ast
import datetime as dt
import hashlib
import importlib.util
import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import atomic_json


def config_keyword(source: str, assignment: str, keyword: str):
    """Read a literal dict-call keyword without evaluating a config or its bases."""
    node, = [n.value for n in ast.parse(source).body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == assignment for t in n.targets)]
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "dict":
        raise ValueError("Expected a dict-call configuration")
    value, = [k.value for k in node.keywords if k.arg == keyword]
    return ast.literal_eval(value)


def class_methods(source: str, name: str) -> list[str]:
    node, = [n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == name]
    return [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def centered_first_box_coordinate(values: list[float], window: int = 5) -> float:
    """Source-formula dependency witness, not an upstream model or stream replay."""
    if window < 1 or window % 2 != 1 or len(values) < window:
        raise ValueError("A full odd-width segment is required")
    half = window // 2
    return sum(values[min(max(i, 0), len(values) - 1)] for i in range(-half, half + 1)) / window


def main() -> None:
    started = time.perf_counter()
    manifest = json.loads((ROOT / "configs/trackocd_core/frontend_alternatives_source_manifest.json").read_text())
    source_root = ROOT / "outputs/trackocd_core/audit/frontend_alternatives_source"
    records, sources = [], {}
    for record in manifest["files"]:
        relative = Path(record["project"]) / record["path"]
        if relative.is_absolute() or ".." in relative.parts or record["bytes"] > 65536:
            raise ValueError("Unsafe or excessive source selection")
        payload = (source_root / relative).read_bytes()
        if len(payload) != record["bytes"]:
            raise ValueError("Source size mismatch: " + str(relative))
        blob = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
        if blob != record["git_blob_sha1"]:
            raise ValueError("Source blob mismatch: " + str(relative))
        repository = manifest["repositories"][record["project"]]
        records.append({**record, "sha256": hashlib.sha256(payload).hexdigest(),
                        "git_blob_matches": True,
                        "url": repository["repository"] + "/blob/" + repository["commit"] + "/" + record["path"]})
        sources[str(relative)] = payload.decode("utf-8")
    cfg = sources["masa/configs/masa-sam/open_vocabulary_mot_test/masa_sam_vitb_open_vocabulary_test.py"]
    sam = sources["masa/masa/models/detectors/sam_masa.py"]
    mot = sources["masa/masa/models/mot/masa.py"]
    post = sources["masa/demo/utils.py"]
    demo = sources["masa/demo/video_demo_with_text.py"]
    aed = sources["aed/datasets/tao_dataset.py"]
    if config_keyword(cfg, "model", "load_public_dets") is not True:
        raise ValueError("Pinned default public-detection branch changed")
    methods = class_methods(sam, "SamMasa")
    if "predict" in methods or "with_neck" in methods:
        raise ValueError("Reinspect native SAM detector API")
    if not all(s in mot for s in ("self.detector.with_neck", "self.detector.predict(", "self.rpn_head.loss_and_predict", "self.roi_head.loss(")):
        raise ValueError("Reinspect native proposal dispatch")
    if not all(s in post for s in ("(half_window, half_window)", "np.convolve", "np.mean(segment_scores)", "invalid_instance_ids.add")):
        raise ValueError("Reinspect offline postprocessing formulas")
    if "if not args.no_post:" not in demo or aed.count("if base_only and cats[ann['category_id']]['frequency'] == 'r'") != 1:
        raise ValueError("Reinspect demo switch or AED base filter")
    plan = json.loads((ROOT / "configs/trackocd_core/masa_sam_candidate_smoke.json").read_text())
    result = {
        "schema_version": "trackocd.core.frontend_alternatives_preflight.v1",
        "audited_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "PASS_SOURCE_PREFLIGHT_NOT_FRONTEND_QUALIFICATION",
        "source_files": records,
        "source_payload_bytes": sum(r["bytes"] for r in records),
        "masa": {
            "candidate": "SAM-B image-only released object-distillation heads plus past-only anonymous association",
            "status": "CONDITIONAL_CANDIDATE_RUNTIME_AND_NATIVE_PROPOSAL_WIRING_UNVERIFIED",
            "documented_supervision": "SA-1B 500K raw images and SAM segments; model zoo reports no in-domain images",
            "released_tensor_coverage_or_complete_training_binding_inspected": False,
            "default_tao_route_uses_external_detic_detections": True,
            "default_public_detections_no_novel_vocabulary_qualified": False,
            "public_detector_vocabulary_or_release_binding_inspected": False,
            "default_benchmark_is_not_our_native_image_only_result": True,
            "sam_class_bases": [ast.unparse(n) for c in ast.parse(sam).body if isinstance(c, ast.ClassDef) and c.name == "SamMasa" for n in c.bases],
            "sam_class_methods": methods,
            "no_public_no_given_default_branch_dispatches_detector_api_not_masa_proposal_heads": True,
            "simple_load_public_dets_false_is_not_a_verified_native_runner": True,
            "demo_offline_post": {
                "default_enabled": True,
                "can_disable_with_no_post": True,
                "symmetric_box_smoothing": True,
                "full_segment_score_mean": True,
                "whole_track_removal_from_later_giant_box": True,
                "causal_candidate_must_bypass_all_three": True,
                "formula_dependency_witness": {
                    "kind": "synthetic source-formula witness, not upstream execution or actual stream",
                    "first_coordinate_before": centered_first_box_coordinate([0., 0., 0., 0., 0.]),
                    "first_coordinate_after_changing_only_future_index_2": centered_first_box_coordinate([0., 0., 10., 0., 0.])}},
            "weight_repository_revision": plan["weight_repository_revision"],
            "weights_downloaded": False,
            "conditional_weight_ceiling_bytes": plan["conditional_weight_ceiling_bytes"],
            "bounded_plan": "configs/trackocd_core/masa_sam_candidate_smoke.json"},
        "aed": {
            "status": "DEFAULT_RELEASE_NOT_DROP_IN_LEGAL_FRONTEND",
            "configured_train_base": True,
            "source_filter": "Exclude LVIS frequency r, not an explicit whitelist of inherited 78 Known IDs",
            "true_protocol_role_overlap_checked": False,
            "released_training_source_weight_binding_verified": False,
            "forbidden_supervision_proven_for_release": False,
            "default_public_detector_vocab_and_supervision_need_audit": True,
            "weight_or_joint_annotation_pack_downloaded": False},
        "runtime_preflight": {
            "existing_environment_root_modules": {n: importlib.util.find_spec(n) is not None
                                                  for n in ("mmengine", "mmcv", "mmdet", "segment_anything", "transformers", "einops", "pycocotools", "timm")},
            "official_cp310_mmcv_2_1_cu118_torch_2_1_wheel_index_observed": True,
            "official_cu118_torch_2_6_index_http_status_observed": 404,
            "compiler_cuda_version_observed": "13.2",
            "new_environment_installed": False,
            "binary_wheel_resolution_and_measured_runtime_size_complete": False},
        "qualification": {"primary_frontend_selected": False, "m1_complete": False,
                          "next": "Resolve minimal binary wheels and checkpoint key coverage before conditional <=8-image Train-only native smoke; no full job"},
        "resources": {"workers": 1, "elapsed_seconds": time.perf_counter() - started,
                      "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
        "boundary": {"upstream_import_or_exec": False, "dataset_or_gt_opened": False,
                     "checkpoint_opened": False, "model_training_or_inference_started": False,
                     "base_environment_changed": False, "test_data_opened": False,
                     "foreign_process_interference": False, "new_representation_correction": False}}
    destination = ROOT / "outputs/trackocd_core/audit/frontend_alternatives_preflight.json"
    atomic_json(destination, result)
    print(json.dumps({"output": str(destination), "source_files": len(records),
                      "source_bytes": result["source_payload_bytes"], "qualification": result["qualification"],
                      "resources": result["resources"]}, indent=2))


if __name__ == "__main__":
    main()
