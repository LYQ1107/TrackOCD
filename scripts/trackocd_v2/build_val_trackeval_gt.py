#!/usr/bin/env python3
"""Materialize the class-agnostic TAO Validation GT for TrackEval.

This adapter is Val-only.  It reads the canonical Validation annotation,
clears category-aware video exclusion fields, maps every GT category to the
single TrackEval foreground class ``1``, and writes a versioned evaluator
input under the v2 output root.  It never references or opens TAO Test.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.trackocd_v2.io import OUTPUT_TARGET, atomic_json, ensure_output_layout, sha256_file


VAL_ANNOTATION = ROOT / "data/raw/tao/annotations/validation.json"
OUTPUT_JSON = OUTPUT_TARGET / "manifests/tao_val_trackeval/validation.json"
OUTPUT_AUDIT = OUTPUT_TARGET / "audit/val_trackeval_gt.json"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def class_agnostic(annotation: dict[str, Any]) -> dict[str, Any]:
    """Convert a TAO annotation object to the pinned single-class schema."""

    videos = []
    for source in annotation["videos"]:
        video = dict(source)
        video["neg_category_ids"] = []
        video["not_exhaustive_category_ids"] = []
        videos.append(video)
    return {
        "info": dict(annotation.get("info") or {}),
        "licenses": list(annotation.get("licenses") or []),
        "videos": videos,
        "images": [dict(source) for source in annotation["images"]],
        "annotations": [
            {**dict(source), "category_id": 1} for source in annotation["annotations"]
        ],
        "tracks": [{**dict(source), "category_id": 1} for source in annotation["tracks"]],
        "categories": [{"id": 1, "name": "object"}],
    }


def build() -> dict[str, Any]:
    if not VAL_ANNOTATION.is_file():
        raise FileNotFoundError(VAL_ANNOTATION)
    source = json.loads(VAL_ANNOTATION.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise ValueError("Validation annotation is not a JSON object")
    for field in ("videos", "images", "annotations", "tracks"):
        if not isinstance(source.get(field), list):
            raise ValueError(f"Validation annotation has no list field: {field}")
    payload = class_agnostic(source)
    atomic_json(OUTPUT_JSON, payload)
    audit = {
        "schema_version": "trackocd.v2.val_trackeval_gt.v1",
        "status": "COMPLETE",
        "generated_utc": _now(),
        "source_annotation": str(VAL_ANNOTATION.resolve()),
        "source_annotation_sha256": sha256_file(VAL_ANNOTATION),
        "output": str(OUTPUT_JSON.resolve()),
        "output_sha256": sha256_file(OUTPUT_JSON),
        "counts": {key: len(payload[key]) for key in ("videos", "images", "annotations", "tracks")},
        "class_agnostic_mapping": {
            "category_id": 1,
            "category_name": "object",
            "source_categories_discarded_from_evaluation": True,
            "video_negative_and_not_exhaustive_categories_cleared": True,
        },
        "split": "val",
        "test_semantic_accessed": False,
        "test_selection_or_tuning": False,
    }
    atomic_json(OUTPUT_AUDIT, audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    ensure_output_layout()
    print(json.dumps(build(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
