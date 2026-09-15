#!/usr/bin/env python3
"""Audit the canonical TAO universe without performing Test semantics.

Test files are opened only far enough to record structural counts and hashes.
No Test category intersection, model selection, threshold search, or semantic
label export is performed here.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.trackocd_v2.io import atomic_json, ensure_output_layout, sha256_file  # noqa: E402
from src.trackocd_v2.protocol import load_ids, validate_roles  # noqa: E402


TRAIN = ROOT / "data/raw/tao/annotations/train.json"
VAL = ROOT / "data/raw/tao/annotations/validation.json"
TAO_TEST_RAW = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal/annotations/tao_test_annotations.json")
TEMPO_TEST = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/tao/annotations/tao_test_lvis_v1_classes.json")
TEMPO_TEST_FINAL = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/tao/annotations/tao_test_lvis_v1_classes_final.json")
KNOWN = ROOT / "data/tao_ow_ocd_v1/splits/known_ids.json"
UNKNOWN_VAL = ROOT / "data/tao_ow_ocd_v1/splits/unknown_ids_val.json"
DISTRACTOR = ROOT / "data/tao_ow_ocd_v1/splits/distractor_ids.json"


def _json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected object at {path}")
    return value


def _counts(value: dict[str, Any]) -> dict[str, int]:
    result = {}
    for key in ("videos", "images", "annotations", "categories", "tracks"):
        values = value.get(key)
        if not isinstance(values, list):
            raise ValueError(f"missing list {key}")
        result[key] = len(values)
    return result


def _file_record(path: Path, *, structural_only: bool) -> dict[str, Any]:
    value = _json(path)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "schema": sorted(value.keys()),
        "counts": _counts(value),
        "category_id_min": min((int(x["id"]) for x in value.get("categories", [])), default=None),
        "category_id_max": max((int(x["id"]) for x in value.get("categories", [])), default=None),
        "structural_only": structural_only,
    }


def _split_record(path: Path, values: set[int]) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "count": len(values), "ids": sorted(values)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    for path in (TRAIN, VAL, TAO_TEST_RAW, TEMPO_TEST, TEMPO_TEST_FINAL, KNOWN, UNKNOWN_VAL, DISTRACTOR):
        if not path.exists():
            raise FileNotFoundError(path)

    known = load_ids(KNOWN)
    novel = load_ids(UNKNOWN_VAL)
    distractor = load_ids(DISTRACTOR)
    validate_roles(known, novel, distractor)
    train = _json(TRAIN)
    train_categories = {int(row["category_id"]) for row in train["annotations"]}
    supported_known = known & train_categories
    zero_shot_known = known - train_categories

    out = ensure_output_layout()
    result = {
        "schema_version": "trackocd.v2.canonical_tao_universe.v1",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_head": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "train_annotation": _file_record(TRAIN, structural_only=False),
        "val_annotation": _file_record(VAL, structural_only=False),
        "test_annotation_lineage": {
            "tempo_track_primary": _file_record(TEMPO_TEST, structural_only=True),
            "official_tao_raw": _file_record(TAO_TEST_RAW, structural_only=True),
            "tempo_track_final_variant": _file_record(TEMPO_TEST_FINAL, structural_only=True),
            "selected_future_final_source": str(TEMPO_TEST.resolve()),
            "semantic_accessed": False,
            "selection_allowed_before_final_freeze": False,
        },
        "category_role_definition": {
            "source_family": "TAO-OW local frozen role files; inherited as protocol source, not fit-known",
            "known_base": _split_record(KNOWN, known),
            "genuine_novel": _split_record(UNKNOWN_VAL, novel),
            "distractor": _split_record(DISTRACTOR, distractor),
            "train_supported_known_count": len(supported_known),
            "train_zero_shot_known_count": len(zero_shot_known),
            "train_supported_known_ids": sorted(supported_known),
            "train_zero_shot_known_ids": sorted(zero_shot_known),
        },
        "protocol": {
            "train_role": "training plus legal known-only semantic supervision",
            "val_role": "development, model/threshold/frontend selection",
            "test_role": "frozen final evaluation only",
            "old_definition": "TAO-OW known/base category set",
            "new_definition": "TAO-OW genuine unknown category set; no fit-known substitution",
            "distractor_policy": "excluded from Old/New headline and retained for diagnostic accounting",
            "test_label_lock": "only path/schema/hash/count structural validation before FINAL_FREEZE",
        },
        "integrity": {
            "role_sets_disjoint": True,
            "train_categories": len(train_categories),
            "test_semantic_gt_used_for_selection": False,
        },
    }
    destination = args.out or (out / "audit/canonical_tao_universe.json")
    atomic_json(destination, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
