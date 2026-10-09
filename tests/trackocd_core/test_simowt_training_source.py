"""Small source-only provenance checks; no upstream runtime or annotations."""
import json
from pathlib import Path

import pytest

from scripts.trackocd_core.audit_simowt_training_source import (
    generator_filter_targets, generator_main_calls, known_ids_in_generator, mapper_path_example,
)

ROOT = Path(__file__).resolve().parents[2]


def test_known_ids_are_extracted_from_generator_not_unrelated_diagnostic():
    source = "knowns = {99}\ndef gen():\n    knowns = {4, 13}\n"
    assert known_ids_in_generator(source) == {4, 13}


def test_local_filter_is_not_reported_as_filtering_saved_annotations():
    local = "def gen():\n    annos = [a for a in annos if a['category_id'] in knowns]\n"
    saved = "def gen():\n    data['annotations'] = [a for a in data['annotations'] if a['category_id'] in knowns]\n"
    assert generator_filter_targets(local) == ["annos"]
    assert generator_filter_targets(saved) == ["data['annotations']"]


def test_commented_entrypoint_is_not_an_executed_training_chain():
    source = "if __name__ == '__main__':\n    view_tao_json()\n    # gen()\n"
    assert generator_main_calls(source) == ["view_tao_json"]


@pytest.mark.parametrize("filename,expected", [
    ("datasets/coco/train2017/000000123456.jpg", "datasets/coco/train2017/000000123456.jpg"),
    ("datasets/coco/train2017/train/source/video/frame.jpg", "datasets/tao/frames/train/source/video/frame.jpg"),
    ("datasets/coco/train2017/source/video/frame.jpg", "datasets/tao/frames/train/source/video/frame.jpg"),
])
def test_source_path_routing_can_reach_train_tao_without_opening_image(filename, expected):
    assert mapper_path_example(filename) == expected


def test_source_trace_does_not_certify_or_condemn_checkpoint_from_auxiliary_file():
    receipt = json.loads((ROOT / "outputs/trackocd_core/audit/simowt_official_training_source.json").read_text())
    assert len(receipt["source_files"]) == 10
    assert sum(r["bytes"] for r in receipt["source_files"]) == receipt["source_payload_bytes"] == 169655
    assert all(r["git_blob_matches"] for r in receipt["source_files"])
    assert receipt["supervision_source_trace"]["known_id_sets_match_inherited_78"]
    assert receipt["training_mapper"]["tao_train_path_rewrite_reachable"]
    assert not receipt["training_mapper"]["mixed_actual_training_file_content_observed"]
    assert receipt["supervision_source_trace"]["gen_extra_anno"]["saved_data_retains_original_annotation_list"]
    gate = receipt["qualification"]
    assert gate["legal_status"] == "BLOCKED_PROVENANCE_NOT_PROVEN_LEAKAGE"
    assert gate["released_checkpoint_supervision_exclusion"] == "UNVERIFIED"
    assert not gate["positive_evidence_of_forbidden_supervision_for_released_weight"]
    assert not gate["primary_frontend_selected"]
    assert not any(receipt["boundary"].values())
    current = json.loads((ROOT / "outputs/trackocd_core/audit/nas_simowt_provenance.json").read_text())
    idol = next(r for r in receipt["source_files"] if r["path"] == "projects/IDOL/idol/idol.py")
    assert idol["sha256"] != current["source_repository"]["current_idol_sha256"]
    compared = {r["path"]: r for r in receipt["nas_current_source_comparison"]["comparisons"]}
    assert not compared["projects/IDOL/idol/idol.py"]["bytes_and_sha256_match"]
    assert not compared["projects/IDOL/train_net.py"]["bytes_and_sha256_match"]
    assert not compared["detectron2/data/datasets/builtin.py"]["bytes_and_sha256_match"]
    assert compared["projects/IDOL/configs/r50_train.yaml"]["bytes_and_sha256_match"]
