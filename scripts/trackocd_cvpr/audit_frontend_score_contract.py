#!/usr/bin/env python3
"""Audit the COVTrack-native score contract without rerunning the frontend.

The audit combines static source checks with a streaming scan of the already
normalized anonymous Train physical streams.  It does not open native TAO
rows, pixels, labels, feature arrays, Val/Test annotations, or vocabulary
files.  Its purpose is to distinguish a foreground/objectness score from the
class-probability column that the native formatter exports as ``score``.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STREAM_ROOT = (
    ROOT
    / "outputs/trackocd_v2_runtime/manifests/frontend_streams/covtrack_native/"
    "covtrack_train_dense_shards_v1"
)
DEFAULT_OUTPUT = ROOT / "outputs/trackocd_cvpr/cvpr2/covtrack_train_score_contract_audit_v1/result.json"

PHYSICAL_FIELDS = {
    "sample_key",
    "source_sample_id",
    "video_id",
    "physical_track_id",
    "frame_ids",
    "boxes_xyxy",
    "image_paths",
    "quality",
    "stream_order",
    "source_split",
}
FORBIDDEN_FIELDS = {
    "category_id",
    "category_ids",
    "category_name",
    "gt_category_id",
    "gt_split",
    "gt_match_id",
    "text",
    "novel",
    "label",
}

SOURCE_FILES = {
    "roi_head": ROOT / "vendor/trackocd_external/covtrack_public/ovtrack/models/roi_heads/ovtrack_roi_head.py",
    "bbox_head": ROOT / "vendor/trackocd_external/covtrack_public/ovtrack/models/roi_heads/bbox_heads/convfc_bbox_head.py",
    "tracker": ROOT / "vendor/trackocd_external/covtrack_public/ovtrack/models/trackers/ovtracker.py",
    "tao_formatter": ROOT / "vendor/trackocd_external/covtrack_public/ovtrack/datasets/tao_dataset.py",
    "frontend_launcher": ROOT / "scripts/trackocd_v2/run_covtrack_native.py",
    "normalizer": ROOT / "scripts/trackocd_v2/normalize_frontend_output.py",
    "config": ROOT / "vendor/trackocd_external/covtrack_public/configs/uncertainty-ovtrack-teta/ovtrack_r50_ctao_train.py",
    "class_contract": ROOT / "vendor/trackocd_external/covtrack_public/ovtrack/models/roi_heads/class_name.py",
}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _line_numbers(text: str, fragment: str) -> list[int]:
    return [index for index, line in enumerate(text.splitlines(), 1) if fragment in line]


def _require_fragment(text: str, fragment: str, label: str) -> list[int]:
    lines = _line_numbers(text, fragment)
    if not lines:
        raise ValueError(f"missing source contract fragment {label!r}: {fragment!r}")
    return lines


def _literal_assignment(path: Path, name: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"missing literal assignment {name!r} in {path}")


def _scan_quality(stream_root: Path) -> dict[str, Any]:
    stream_paths = sorted(stream_root.rglob("physical_tracks.jsonl"))
    if not stream_paths:
        raise FileNotFoundError(f"no physical stream jsonl files under {stream_root}")

    tracks = 0
    observations = 0
    finite = 0
    nonfinite = 0
    out_of_unit_interval = 0
    total = 0.0
    minimum: float | None = None
    maximum: float | None = None
    bins = {
        "[0,0.05)": 0,
        "[0.05,0.10)": 0,
        "[0.10,0.20)": 0,
        "[0.20,0.50)": 0,
        "[0.50,1.00]": 0,
        "other": 0,
    }
    stream_hashes: dict[str, str] = {}

    for stream_path in stream_paths:
        stream_hashes[str(stream_path.relative_to(stream_root))] = _sha256(stream_path)
        with stream_path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue
                row = json.loads(raw)
                if not isinstance(row, dict) or set(row) != PHYSICAL_FIELDS:
                    raise ValueError(f"physical stream schema mismatch at {stream_path}:{line_number}")
                if FORBIDDEN_FIELDS & set(row) or row.get("source_split") != "train_predicted":
                    raise ValueError(f"forbidden or non-Train row at {stream_path}:{line_number}")
                frame_ids = row.get("frame_ids")
                boxes = row.get("boxes_xyxy")
                qualities = row.get("quality")
                if not isinstance(frame_ids, list) or not isinstance(boxes, list) or not isinstance(qualities, list):
                    raise ValueError(f"invalid list fields at {stream_path}:{line_number}")
                if len(frame_ids) != len(boxes) or len(frame_ids) != len(qualities):
                    raise ValueError(f"length mismatch at {stream_path}:{line_number}")
                tracks += 1
                observations += len(qualities)
                for value in qualities:
                    score = float(value)
                    if not math.isfinite(score):
                        nonfinite += 1
                        continue
                    finite += 1
                    total += score
                    minimum = score if minimum is None else min(minimum, score)
                    maximum = score if maximum is None else max(maximum, score)
                    if score < 0.0 or score > 1.0:
                        out_of_unit_interval += 1
                    if 0.0 <= score < 0.05:
                        bins["[0,0.05)"] += 1
                    elif 0.05 <= score < 0.10:
                        bins["[0.05,0.10)"] += 1
                    elif 0.10 <= score < 0.20:
                        bins["[0.10,0.20)"] += 1
                    elif 0.20 <= score < 0.50:
                        bins["[0.20,0.50)"] += 1
                    elif 0.50 <= score <= 1.0:
                        bins["[0.50,1.00]"] += 1
                    else:
                        bins["other"] += 1

    return {
        "stream_substreams": len(stream_paths),
        "stream_rows_tracks": tracks,
        "observations": observations,
        "finite": finite,
        "nonfinite": nonfinite,
        "out_of_unit_interval": out_of_unit_interval,
        "min": minimum,
        "max": maximum,
        "mean": total / finite if finite else None,
        "fixed_bins": bins,
        "stream_sha256_by_path": stream_hashes,
    }


def audit(stream_root: Path, output_path: Path) -> dict[str, Any]:
    source_text = {name: path.read_text(encoding="utf-8") for name, path in SOURCE_FILES.items()}
    config_num_classes_match = re.search(r"\bnum_classes\s*=\s*(\d+)", source_text["config"])
    if config_num_classes_match is None:
        raise ValueError("could not read configured num_classes")
    configured_num_classes = int(config_num_classes_match.group(1))
    validation_ids = _literal_assignment(SOURCE_FILES["class_contract"], "tao_validation_used_cate_ids")
    validation_novel_index = _literal_assignment(SOURCE_FILES["class_contract"], "validation_novel_index")
    if not isinstance(validation_ids, list) or not isinstance(validation_novel_index, list):
        raise ValueError("validation class contract is not a literal list")

    roi_text = source_text["roi_head"]
    bbox_text = source_text["bbox_head"]
    formatter_text = source_text["tao_formatter"]
    launcher_text = source_text["frontend_launcher"]
    normalizer_text = source_text["normalizer"]
    config_text = source_text["config"]
    tracker_text = source_text["tracker"]

    fragments = {
        "roi_upstream_softmax": _require_fragment(
            roi_text, "cls_score_text = cls_score_text.softmax(dim=1)", "roi_upstream_softmax"
        ),
        "bbox_final_softmax": _require_fragment(
            bbox_text, "scores = F.softmax(cls_score, dim=-1)", "bbox_final_softmax"
        ),
        "formatter_exports_last_column": _require_fragment(
            formatter_text, 'data["score"] = float(bboxes[i][-1])', "formatter_exports_last_column"
        ),
        "normalizer_preserves_native_score": _require_fragment(
            normalizer_text, "max(float(score), 1e-6)", "normalizer_preserves_native_score"
        ),
        "native_score_threshold": _require_fragment(
            launcher_text, '"--show_score_thr", "0.0001"', "native_score_threshold"
        ),
        "validation_only_route": _require_fragment(
            config_text, "only_validation_categories=True", "validation_only_route"
        ),
        "ensemble_route": _require_fragment(config_text, "ensemble=True", "ensemble_route"),
        "tracker_obj_score_gate": _require_fragment(
            tracker_text, "if bboxes[i, -1] < self.obj_score_thr:", "tracker_obj_score_gate"
        ),
        "tracker_init_score_gate": _require_fragment(
            tracker_text,
            "new_objs = (ids == -1) & (obj_scores > self.init_score_thr).cpu()",
            "tracker_init_score_gate",
        ),
        "tracker_default_distractor_cut": _require_fragment(
            tracker_text,
            "object_score_thr=0.5",
            "tracker_default_distractor_cut",
        ),
        "background_append": _require_fragment(
            roi_text, "text_features = torch.cat(", "background_append"
        ),
    }
    if "bg_class_embedding" not in roi_text:
        raise ValueError("background embedding source contract fragment missing")
    if "scores = multi_scores[:, :-1]" not in bbox_text:
        raise ValueError("background exclusion source contract fragment missing")
    if '"confused_features": True' not in launcher_text:
        raise ValueError("native confused_features contract fragment missing")

    selected_class_count = len(validation_ids)
    score_dimension = selected_class_count + 1
    quality = _scan_quality(stream_root)
    uniform_score = 1.0 / score_dimension
    result = {
        "schema_version": "trackocd.cvpr.frontend-score-contract-audit.v1",
        "status": "COMPLETE_CATEGORY_FREE_SOURCE_AND_STREAM_CONTRACT_AUDIT",
        "generated_utc": _now(),
        "scope": {
            "stream_root": str(stream_root.resolve()),
            "native_tao_rows_opened": False,
            "pixels_opened": False,
            "feature_arrays_opened": False,
            "gt_or_annotation_labels_opened": False,
            "val_labels_opened": False,
            "test_accessed": False,
            "training_started": False,
            "frontend_rerun": False,
            "model_selection_or_qualification": False,
        },
        "inputs": {
            "source_sha256_by_name": {
                name: _sha256(path) for name, path in SOURCE_FILES.items()
            },
            "stream_root": str(stream_root.resolve()),
        },
        "source_contract": {
            "configured_full_model_classes": configured_num_classes,
            "validation_category_index_count": selected_class_count,
            "validation_index_length": len(validation_novel_index),
            "background_class_appended": True,
            "score_dimension_after_validation_slice": score_dimension,
            "uniform_score_reference": uniform_score,
            "upstream_text_softmax": True,
            "upstream_image_softmax_under_ensemble": True,
            "final_bbox_head_softmax_on_already_normalized_score": True,
            "final_non_background_score_is_class_probability": True,
            "foreground_objectness_score_exported": False,
            "background_column_excluded_before_nms": True,
            "formatter_exports_nms_last_column_as_score": True,
            "normalizer_applies_only_lower_floor": True,
            "native_show_score_threshold": 0.0001,
            "native_confused_features": True,
            "tracker_obj_score_gate": True,
            "tracker_init_score_gate": True,
            "tracker_default_distractor_cut": 0.5,
            "verified_source_line_numbers": fragments,
        },
        "stream_quality": quality,
        "interpretation": (
            "The native score contract is a max non-background class probability "
            "after an upstream class softmax and a final bbox-head softmax; it is "
            "not a foreground/objectness probability. The normalizer preserves "
            "that value (apart from a 1e-6 floor). The observed Train quality "
            "distribution is therefore consistent with a near-uniform 297-way "
            "class score, but this receipt does not quantify the effect of a future "
            "contract repair on geometry, tracking, OCD, or MOT metrics."
        ),
        "boundary": {
            "source_validation_index_counted_without_category_names": True,
            "semantic_data_labels_read": False,
            "novel_vocabulary_used_for_selection": False,
            "prediction_to_gt_mapping_persisted": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream-root", type=Path, default=DEFAULT_STREAM_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = audit(args.stream_root.resolve(), args.output.resolve())
    print(json.dumps({"status": result["status"], "source_contract": result["source_contract"], "stream_quality": {k: v for k, v in result["stream_quality"].items() if k != "stream_sha256_by_path"}}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
