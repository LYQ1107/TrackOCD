#!/usr/bin/env python3
"""Materialize the frozen, category-free TAO Test GT for TrackEval.

The Test annotation is intentionally opened only after ``FINAL_FREEZE.json``
has passed the protocol guard.  TrackEval's TAO-OW implementation expects one
JSON file in a GT directory, so this post-freeze adapter preserves the TAO
video/image/track lineage while mapping every GT category to the single
class-agnostic foreground class ``1``.
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

from src.trackocd_v2.io import OUTPUT_TARGET, atomic_json, ensure_output_layout, sha256_file  # noqa: E402
from src.trackocd_v2.protocol import assert_test_semantic_access_allowed  # noqa: E402


TEST_ANNOTATION = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/tao/annotations/tao_test_lvis_v1_classes.json"
)
FINAL_FREEZE = OUTPUT_TARGET / "audit/FINAL_FREEZE.json"
OUTPUT_JSON = OUTPUT_TARGET / "manifests/tao_test_trackeval/tao_test.json"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _load_annotation() -> dict[str, Any]:
    value = json.loads(TEST_ANNOTATION.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("TAO Test annotation is not an object")
    for field in ("videos", "images", "annotations", "tracks"):
        if not isinstance(value.get(field), list):
            raise ValueError(f"TAO Test annotation has no list field {field}")
    return value


def _class_agnostic(annotation: dict[str, Any]) -> dict[str, Any]:
    videos = []
    for source in annotation["videos"]:
        video = dict(source)
        # These fields are category-aware TAO metadata.  Empty lists retain
        # the TAO-OW schema while making the class-agnostic evaluation
        # independent of the source category vocabulary.
        video["neg_category_ids"] = []
        video["not_exhaustive_category_ids"] = []
        videos.append(video)

    images = [dict(source) for source in annotation["images"]]
    annotations = []
    for source in annotation["annotations"]:
        item = dict(source)
        item["category_id"] = 1
        annotations.append(item)

    tracks = []
    for source in annotation["tracks"]:
        item = dict(source)
        item["category_id"] = 1
        tracks.append(item)

    return {
        "info": dict(annotation.get("info") or {}),
        "licenses": list(annotation.get("licenses") or []),
        "videos": videos,
        "images": images,
        "annotations": annotations,
        "tracks": tracks,
        "categories": [{"id": 1, "name": "object"}],
    }


def build() -> dict[str, Any]:
    # Keep this before the Test source existence check and before annotation
    # parsing so this utility cannot become a pre-freeze semantic accessor.
    assert_test_semantic_access_allowed(FINAL_FREEZE, "build frozen TAO Test TrackEval GT")
    if not TEST_ANNOTATION.is_file():
        raise FileNotFoundError(TEST_ANNOTATION)
    freeze_sha256 = sha256_file(FINAL_FREEZE)
    annotation_sha256 = sha256_file(TEST_ANNOTATION)
    source = _load_annotation()
    payload = _class_agnostic(source)
    atomic_json(OUTPUT_JSON, payload)
    audit = {
        "schema_version": "trackocd.v2.test_trackeval_gt.v1",
        "status": "COMPLETE",
        "generated_utc": _now(),
        "source_annotation": str(TEST_ANNOTATION.resolve()),
        "source_annotation_sha256": annotation_sha256,
        "output": str(OUTPUT_JSON.resolve()),
        "output_sha256": sha256_file(OUTPUT_JSON),
        "counts": {
            "videos": len(payload["videos"]),
            "images": len(payload["images"]),
            "annotations": len(payload["annotations"]),
            "tracks": len(payload["tracks"]),
        },
        "class_agnostic_mapping": {
            "category_id": 1,
            "category_name": "object",
            "source_categories_discarded_from_evaluation": True,
            "video_negative_and_not_exhaustive_categories_cleared": True,
        },
        "final_freeze": str(FINAL_FREEZE.resolve()),
        "final_freeze_sha256": freeze_sha256,
        "test_semantic_accessed": True,
        "test_selection_or_tuning": False,
    }
    atomic_json(OUTPUT_TARGET / "audit/test_trackeval_gt.json", audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    ensure_output_layout()
    result = build()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
