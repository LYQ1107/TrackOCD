"""Build and validate the label-free 152-event Phase74S model contract."""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .io import canonical_hash, sha256


MODEL_FIELDS = ("model_event_uid", "source_tracklet_keys", "target_tracklet_key", "source_video", "target_video")
FORBIDDEN_MODEL_FIELDS = {
    "event_key", "category", "category_id", "category_gt_denominator_only",
    "distractor_category_gt_denominator_only", "kind", "role", "polarity",
    "fold", "expected_first_commit", "target_first_reliable_prefix_index_gt_only",
    "target_row_keys", "raw_hard_negative_similarity", "semantic_id", "physical_id",
}


def _video_from_tracklet(key: str) -> int:
    match = re.fullmatch(r"v(-?\d+):p-?\d+", str(key))
    if match is None:
        raise ValueError(f"malformed tracklet key: {key!r}")
    return int(match.group(1))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    import json
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_model_contract(root: Path, output: Path | None = None) -> dict[str, Any]:
    """Create opaque model rows in frozen evaluator order.

    Source/target videos come from explicit evaluator metadata fields, while
    the model-facing record contains no event key, polarity, category, fold or
    GT fields.  The event key appears only in the evaluator-only join table.
    """
    pos_path = root / "outputs/iclr27_phase19r/manifests/held_known_positive_events.jsonl"
    neg_path = root / "outputs/iclr27_phase19r/manifests/held_known_negative_events.jsonl"
    model_path = root / "outputs/iclr27_phase19r/manifests/held_known_model_events.jsonl"
    positive = _read_jsonl(pos_path)
    negative = _read_jsonl(neg_path)
    model_rows = _read_jsonl(model_path)
    evaluator = [("positive", row) for row in positive] + [("negative", row) for row in negative]
    model_by_key = {str(row["event_key"]): row for row in model_rows}
    if len(model_by_key) != len(model_rows):
        raise ValueError("held_known_model_events has duplicate event keys")
    model_records: list[dict[str, Any]] = []
    join_records: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, (polarity, truth) in enumerate(evaluator):
        event_key = str(truth.get("event_key", ""))
        source_keys = [str(x) for x in truth.get("source_tracklet_keys", [])]
        if not source_keys:
            errors.append(f"{event_key}: missing source tracklet")
            continue
        source_video = int(truth["source_video"])
        target_video = int(truth["target_video"])
        source_from_key = [_video_from_tracklet(key) for key in source_keys]
        if any(video != source_video for video in source_from_key):
            errors.append(f"{event_key}: explicit source_video disagrees with source tracklet")
            continue
        base = model_by_key.get(event_key)
        if base is None:
            errors.append(f"{event_key}: missing model manifest row")
            continue
        if [str(x) for x in base.get("source_tracklet_keys", [])] != source_keys or str(base.get("target_tracklet_key")) != str(truth.get("target_tracklet_key")):
            errors.append(f"{event_key}: model/evaluator track fields differ")
            continue
        if int(base.get("target_video", target_video)) != target_video:
            errors.append(f"{event_key}: model/evaluator target_video differs")
            continue
        uid = f"evt_{index:06d}"
        record = {
            "model_event_uid": uid,
            "source_tracklet_keys": source_keys,
            "target_tracklet_key": str(truth["target_tracklet_key"]),
            "source_video": source_video,
            "target_video": target_video,
        }
        model_records.append(record)
        join_records.append({
            "model_event_uid": uid,
            "event_key": event_key,
            "fold": int(truth["fold"]),
            "kind": str(truth["kind"]),
            "polarity": polarity,
            "category": truth.get("category_gt_denominator_only") if polarity == "positive" else truth.get("target_category_gt_denominator_only"),
            "target_first_reliable_prefix_index_gt_only": int(truth["target_first_reliable_prefix_index_gt_only"]),
            "source_tracklet_keys": source_keys,
            "target_tracklet_key": str(truth["target_tracklet_key"]),
            "source_video": source_video,
            "target_video": target_video,
        })
    if errors:
        raise ValueError("; ".join(errors[:8]))
    if len(model_records) != 152 or len(join_records) != 152:
        raise ValueError(f"expected 152 rows, got model={len(model_records)} join={len(join_records)}")
    model_keys = {row["model_event_uid"] for row in model_records}
    event_keys = {row["event_key"] for row in join_records}
    if len(model_keys) != 152 or len(event_keys) != 152:
        raise ValueError("v2 contract has duplicate opaque UIDs or event keys")
    forbidden_seen = sorted({key for row in model_records for key in row if key in FORBIDDEN_MODEL_FIELDS})
    contract = {
        "schema_version": "phase74s.model_evaluator_contract.v2",
        "model_manifest_fields": list(MODEL_FIELDS),
        "model_manifest_forbidden_fields": sorted(FORBIDDEN_MODEL_FIELDS),
        "model_event_count": len(model_records),
        "evaluator_event_count": len(join_records),
        "join_count": len(join_records),
        "model_uid_unique": len(model_keys) == 152,
        "evaluator_event_key_unique": len(event_keys) == 152,
        "missing_model": 0,
        "missing_evaluator": 0,
        "duplicate_model": 152 - len(model_keys),
        "duplicate_evaluator": 152 - len(event_keys),
        "fold_counts": {str(k): int(v) for k, v in sorted(Counter(row["fold"] for row in join_records).items())},
        "polarity_counts": dict(Counter(row["polarity"] for row in join_records)),
        "model_order_sha256": canonical_hash(model_records),
        "join_order_sha256": canonical_hash(join_records),
        "event_key_order_sha256": canonical_hash([row["event_key"] for row in join_records]),
        "forbidden_model_fields_seen": forbidden_seen,
        "source_video_derivation": "explicit evaluator source_video cross-checked against source_tracklet key; no event_key parsing",
        "labels_joined_after_model_contract": True,
        "model_input_contains_evaluator_labels": False,
        "causal_order": "positive file order followed by negative file order, inherited frozen evaluator order",
        "protocol_hash_inputs": {
            "positive_manifest_sha256": sha256(pos_path),
            "negative_manifest_sha256": sha256(neg_path),
            "held_model_manifest_sha256": sha256(model_path),
        },
    }
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
    return {"model_records": model_records, "join_records": join_records, "contract": contract}
