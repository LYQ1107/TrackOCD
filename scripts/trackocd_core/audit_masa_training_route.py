#!/usr/bin/env python3
"""Pinned published training-route AST audit, not release-stage certification."""
from __future__ import annotations
import ast
import datetime as dt
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_core.masa_native import assignment
from src.trackocd_v2.io import atomic_json, sha256_file
SOURCE = ROOT / "outputs/trackocd_core/audit/frontend_alternatives_source/masa"


def inspect_route(config: str, converter: str, dataset: str) -> dict:
    dataset_type = assignment(config, "dataset_type", {})
    scale = assignment(config, "img_scale", {})
    pipeline = assignment(config, "train_pipeline", {"img_scale": scale})
    loader = assignment(config, "train_dataloader", {"dataset_type": dataset_type, "train_pipeline": pipeline})
    leaves = loader["dataset"]["dataset"]["datasets"]
    tree = ast.parse(converter)
    process, = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "process_file"]
    categories = []
    for node in ast.walk(process):
        if isinstance(node, ast.Dict):
            categories.extend(ast.literal_eval(v) for k, v in zip(node.keys, node.values)
                              if isinstance(k, ast.Constant) and k.value == "category_id")
    cls, = [n for n in ast.parse(dataset).body if isinstance(n, ast.ClassDef) and n.name == "MASADataset"]
    meta, = [ast.literal_eval(n.value) for n in cls.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "METAINFO" for t in n.targets)]
    return {"training_dataset_leaves": [{k: leaf[k] for k in ("type", "ann_file", "data_prefix")} for leaf in leaves],
            "training_fixed_length": loader["dataset"]["dataset"]["fixed_length"],
            "converted_category_ids": categories,
            "generic_object_metadata_only": meta["classes"] in ("object", ("object",)),
            "all_selected_leaf_paths_are_sa1b": all(leaf["ann_file"] == "data/sam/sam_annotations/jsons/sa1b_coco_fmt_500k_bbox_anno.json"
                                                   and leaf["data_prefix"]["img"] == "data/sam/batch0/" for leaf in leaves),
            "converter_assigns_one_generic_category": categories == [1, 1],
            "scope": "Published GroundingDINO training template/data converter; not a SAM-B checkpoint training config"}


def main() -> int:
    started = time.monotonic()
    manifest_path = ROOT / "configs/trackocd_core/masa_training_source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    records = []
    for record in manifest["files"]:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts or record["bytes"] > 16000:
            raise ValueError("Unexpected small source path/size")
        path = SOURCE / relative
        data = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if path.is_symlink() or len(data) != record["bytes"] or blob != record["git_blob_sha1"]:
            raise ValueError("Pinned training-source identity mismatch")
        records.append({**record, "sha256": hashlib.sha256(data).hexdigest()})
    route = inspect_route((SOURCE / "configs/datasets/masa_dataset.py").read_text(),
                          (SOURCE / "tools/format_conversion/convert_sam_2_cocofmt.py").read_text(),
                          (SOURCE / "masa/datasets/masa_dataset.py").read_text())
    zoo = (SOURCE / "docs/model_zoo.md").read_text()
    train = (SOURCE / "docs/train.md").read_text()
    if ("https://huggingface.co/dereksiyuanli/masa/resolve/main/sam_vitb_masa.pth" not in zoo
            or "do not use any in-domain images" not in zoo or "500K images sampled from SA-1B" not in train):
        raise ValueError("Pinned published model-zoo/default-data assertions changed")
    if not all(route[k] for k in ("all_selected_leaf_paths_are_sa1b", "converter_assigns_one_generic_category", "generic_object_metadata_only")):
        raise ValueError("Published generic route differs from inspected assumptions")
    result = {"schema_version": "trackocd.core.masa_training_route.v1",
              "status": "PUBLISHED_GENERIC_ROUTE_VERIFIED_EXACT_SAM_RELEASE_STAGE_BINDING_MISSING",
              "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "source_commit": manifest["commit"],
              "source_files": records, "source_bytes": sum(r["bytes"] for r in records), "published_route": route,
              "official_model_zoo_links_exact_sam_checkpoint_filename": "sam_vitb_masa.pth",
              "checkpoint_sha256": "441c05bf9519632fead1afd5200bb6a4f13b4a41c58f4428024490b4c2bd777c",
              "no_in_domain_training_claim_scope": "Official model-zoo/train docs, not checkpoint-saved source/config/training-data ledger",
              "sam_training_config_paths_in_complete_pinned_tree": manifest["sam_b_training_config_paths"],
              "documented_converter_filename_present": manifest["documented_converter_path_present"],
              "actual_converter_filename": "tools/format_conversion/convert_sam_2_cocofmt.py",
              "exact_checkpoint_stage_config_supervision_binding_verified": False,
              "forbidden_supervision_proven": False, "primary_freeze_permitted_by_this_audit": False,
              "boundary": {"upstream_code_executed": False, "training_json_opened": False, "dataset_or_weight_download": False,
                           "gt_or_test_access": False, "model_execution": False, "foreign_process_interference": False},
              "resources": {"wall_seconds": time.monotonic() - started,
                            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}}
    atomic_json(ROOT / "outputs/trackocd_core/audit/masa_training_route.json", result)
    print(json.dumps({k: result[k] for k in ("status", "source_bytes", "sam_training_config_paths_in_complete_pinned_tree", "documented_converter_filename_present", "resources")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
