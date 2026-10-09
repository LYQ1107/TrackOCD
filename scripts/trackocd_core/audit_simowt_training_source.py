#!/usr/bin/env python3
"""Read-only pinned source trace, not a checkpoint-supervision certificate.

No upstream import/exec, annotation, weights, model, training or inference.
Only the ten allowlisted small source files and the inherited role IDs are read.
"""
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
from src.trackocd_v2.io import atomic_json


def known_ids_in_generator(source: str) -> set[int]:
    module = ast.parse(source)
    function, = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "gen"]
    value, = [n.value for n in function.body if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == "knowns" for t in n.targets)]
    result = ast.literal_eval(value)
    if not isinstance(result, set) or not all(type(v) is int for v in result):
        raise ValueError("Expected a literal set of Known IDs")
    return result


def generator_filter_targets(source: str) -> list[str]:
    """Return assignment destinations of category-membership filters, without executing them."""
    module = ast.parse(source)
    function, = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "gen"]
    targets = []
    for node in function.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.ListComp):
            continue
        if any(isinstance(c, ast.Compare) and isinstance(c.comparators[0], ast.Name)
               and c.comparators[0].id == "knowns"
               and isinstance(c.ops[0], ast.In)
               for gen in node.value.generators for c in gen.ifs):
            targets.extend(ast.unparse(t) for t in node.targets)
    return targets


def generator_main_calls(source: str) -> list[str]:
    module = ast.parse(source)
    guard, = [n for n in module.body if isinstance(n, ast.If)
              and ast.unparse(n.test) == "__name__ == '__main__'"]
    return [n.value.func.id for n in guard.body if isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)]


def mapper_path_example(filename: str) -> str:
    """Copy only inspected string-routing logic; never open an image or call a mapper."""
    if "train2017/0" not in filename:
        filename = filename.replace("coco/train2017", "tao/frames/train")
        if "train/train" in filename:
            filename = filename.replace("/train/train/", "/train/")
    return filename


def main() -> None:
    started = time.perf_counter()
    manifest = json.loads((ROOT / "configs/trackocd_core/simowt_official_source_manifest.json").read_text())
    source_root = ROOT / "outputs/trackocd_core/audit/simowt_official_source"
    records, sources = [], {}
    for record in manifest["files"]:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts or record["bytes"] > 65536:
            raise ValueError("Unsafe or unexpectedly large source selection")
        path = source_root / relative
        if path.stat().st_size != record["bytes"]:
            raise ValueError("Source size mismatch: " + record["path"])
        payload = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
        if blob != record["git_blob_sha1"]:
            raise ValueError("Pinned Git blob mismatch: " + record["path"])
        records.append({**record, "sha256": hashlib.sha256(payload).hexdigest(),
                        "git_blob_matches": True,
                        "url": manifest["repository"] + "/blob/" + manifest["commit"] + "/" + record["path"]})
        sources[record["path"]] = payload.decode("utf-8")

    roles = json.loads((ROOT / "configs/trackocd_core/roles.json").read_text())
    known = set(roles["known_ids"])
    extra, faker = (sources["dataset_extra/" + name + ".py"] for name in ("gen_extra_anno", "gen_faker_anno"))
    if known_ids_in_generator(extra) != known or known_ids_in_generator(faker) != known:
        raise ValueError("Official auxiliary Known IDs differ from inherited protocol")
    if generator_filter_targets(extra) != ["annos"] or generator_filter_targets(faker) != ["gtdata['annotations']"]:
        raise ValueError("Unexpected auxiliary filter destination")
    if generator_main_calls(extra) != ["gen"] or generator_main_calls(faker) != ["view_tao_json"]:
        raise ValueError("Unexpected auxiliary entrypoint")
    registry = ast.parse(sources["detectron2/data/datasets/builtin.py"])
    agn, = [ast.literal_eval(n.value) for n in registry.body if isinstance(n, ast.Assign)
            and any(ast.unparse(t) == "_PREDEFINED_SPLITS_COCO['coco_agn']" for t in n.targets)]
    # These are source filename strings, not annotation files opened by this audit.
    examples = ["datasets/coco/train2017/000000123456.jpg",
                "datasets/coco/train2017/train/source/video/frame.jpg",
                "datasets/coco/train2017/source/video/frame.jpg"]
    # Observations already returned by NAS command exec-aee15cc8-b452-429a-905b-57e38ffdf01d.
    # Do not treat the same Git HEAD as byte-identical dirty worktree source.
    nas_records = {
        "projects/IDOL/idol/idol.py": (58283, "00970901e4501879f4722590a085817a4a49ec6f51683d798a8760e9c948c4bb"),
        "detectron2/data/datasets/builtin.py": (11076, "0844d1ae0733eda67c04325b283c4271d883fa5eac060b20cc2b9b814b8901de"),
        "projects/IDOL/train_net.py": (6977, "fc28312a704e1843b0fab4e59c812f45ddf5bfbda878ca22681ee5f5bf9aa61b"),
        "projects/IDOL/configs/r50_train.yaml": (1174, "6b699c50af57634eb714f2b7fe974e9560ff8db103f3a80e773e0c0a8a3cf2ee"),
    }
    comparisons = [{"path": r["path"], "upstream_bytes": r["bytes"],
                    "upstream_sha256": r["sha256"], "nas_current_bytes": nas_records[r["path"]][0],
                    "nas_current_sha256": nas_records[r["path"]][1],
                    "bytes_and_sha256_match": (r["bytes"], r["sha256"]) == nas_records[r["path"]]}
                   for r in records if r["path"] in nas_records]
    result = {
        "schema_version": "trackocd.core.simowt_official_training_source.v1",
        "audited_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "PASS_STATIC_SOURCE_TRACE_NOT_CHECKPOINT_QUALIFICATION",
        "upstream_commit": manifest["commit"],
        "source_files": records,
        "source_payload_bytes": sum(r["bytes"] for r in records),
        "nas_current_source_comparison": {
            "evidence_thread": "01a0c851-ce6c-76e2-ac44-d6b0e3618ceb",
            "evidence_command": "exec-aee15cc8-b452-429a-905b-57e38ffdf01d",
            "comparisons": comparisons,
            "same_HEAD_proves_same_worktree_source": False,
            "complete_dirty_patch_recovered": False},
        "configured_training": {"dataset": "coco_2017_train_agn", "num_classes": 1,
                                "coco_pretrain": True, "initial_weights": "./weights/R-50.pkl",
                                "base_yaml_inheritance": False},
        "registered_paths": {k: list(agn[k]) for k in ("coco_2017_train_agn", "coco_2017_val_agn")},
        "training_mapper": {
            "source_lines": [121, 124], "tao_train_path_rewrite_reachable": True,
            "two_augmented_views_of_same_source_image": True,
            "path_examples": [{"input": p, "routed": mapper_path_example(p)} for p in examples],
            "mixed_actual_training_file_content_observed": False},
        "supervision_source_trace": {
            "known_id_sets_match_inherited_78": True,
            "gen_faker_anno": {
                "proposal_path_string": "datasets/props_train.json",
                "gt_path_is_named_train": True,
                "gt_annotations_filtered_in_place_to_known": True,
                "merged_foreground_only_and_generated_box_masks": True,
                "save_statements_commented_out": True,
                "default_entrypoint_calls_gen": False,
                "teacher_proposal_supervision_and_merged_json_lineage_verified": False},
            "gen_extra_anno": {
                "gt_path_is_named_train": True,
                "filter_destination": "local annos variable, not data['annotations']",
                "saved_data_retains_original_annotation_list": True,
                "used_in_released_checkpoint_training": "UNVERIFIED"},
            "score_to_loss_path": {
                "loader": "annotation longscore -> int(longscore*1000), default 1000; coco.py:224-227",
                "model": "class modulo 10000 -> longscore/1000; foreground labels zero; idol.py:423-430",
                "criterion": "sqrt(longscore) weights contrastive/aux ReID loss, COCO weights overwritten 1; deformable_detr.py:646-708",
                "category_collapse_does_not_prove_box_supervision_origin": True}},
        "pseudo_generation_trace": {
            "default_eval_function_choice": "2 (tracking), not 1/3 (pseudo generation)",
            "pseudo_generators_exist_but_active_config_and_merge_script_not_sealed": True,
            "documented_nested_eval_config_and_truncode_missing_from_complete_pinned_tree": True},
        "qualification": {
            "released_checkpoint_supervision_exclusion": "UNVERIFIED",
            "historical_run_source_and_weight_binding": "UNVERIFIED",
            "positive_evidence_of_forbidden_supervision_for_released_weight": False,
            "known_only_auxiliary_source_is_not_complete_training_proof": True,
            "primary_frontend_selected": False,
            "legal_status": "BLOCKED_PROVENANCE_NOT_PROVEN_LEAKAGE"},
        "minimal_missing_evidence": [
            "Released checkpoint SHA -> every training/self-training stage config/source and initial/teacher SHA",
            "Hashes and split/source-role summary of actual merged training JSON; Known GT only, pseudo-vs-GT provenance",
            "Teacher proposal and pseudo generation split/supervision/vocabulary lineage plus merge/filter source",
            "Train/Val separation and strict frame-only inference binding; no novel label/vocabulary or Test access"],
        "resources": {"elapsed_seconds": time.perf_counter() - started,
                      "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      "workers": 1},
        "boundary": {"upstream_import_or_exec": False, "annotation_or_model_opened": False,
                     "training_or_inference_started": False, "test_data_opened": False,
                     "source_or_frozen_stream_mutated": False},
    }
    destination = ROOT / "outputs/trackocd_core/audit/simowt_official_training_source.json"
    atomic_json(destination, result)
    print(json.dumps({"output": str(destination), "source_files": len(records),
                      "source_bytes": result["source_payload_bytes"],
                      "qualification": result["qualification"], "resources": result["resources"]}, indent=2))


if __name__ == "__main__":
    main()
