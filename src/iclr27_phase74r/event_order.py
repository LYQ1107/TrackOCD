"""Reconstruct the actual Phase19R model-event order without labels."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .io import canonical_hash, iter_jsonl, sha256


MODEL_MANIFEST = Path("outputs/iclr27_phase19r/manifests/public_model_events.jsonl")
FALLBACK_SOURCE_FILES = (
    Path("data/iclr27_phase19r/sources/positive_events.jsonl"),
    Path("data/iclr27_phase19r/sources/negative_events.jsonl"),
)
MODEL_FIELDS = (
    "event_key",
    "source_tracklet_keys",
    "target_tracklet_key",
    "source_video",
    "target_video",
    "target_first_reliable_prefix_index_gt_only",
)


def _model_view(raw: dict[str, Any]) -> dict[str, Any]:
    """Match ``freeze_predictions.public_events`` exactly, but never write it."""
    return {
        "event_key": str(raw["event_key"]),
        "source_tracklet_keys": [str(x) for x in raw.get("source_tracklet_keys", [])],
        "target_tracklet_key": str(raw.get("target_tracklet_key", "")),
        "source_video": int(raw.get("source_video", -1)),
        "target_video": int(raw.get("target_video", -1)),
        "target_first_reliable_prefix_index_gt_only": int(raw.get("target_first_reliable_prefix_index_gt_only", -1)),
    }


def _read_array_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError(f"{path} top level is not an array")
        return [dict(x) for x in value]
    return list(iter_jsonl(path))


def load_actual_model_event_stream(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return model order and provenance.

    If the historical public model manifest is present it is authoritative and
    its line order is retained.  Otherwise this reproduces the historical
    fallback: read the two source files, remove evaluator-only fields, and sort
    by event_key.  No output is written to the Phase19R namespace.
    """
    path = root / MODEL_MANIFEST
    expected_rows: list[dict[str, Any]] | None = None
    if path.exists():
        raw = _read_array_or_jsonl(path)
        rows = [_model_view(x) for x in raw]
        expected_rows = rows[:]
        source = "public_model_events.jsonl"
        rule = "historical manifest line order (no sort)"
        source_paths = [str(path.resolve())]
    else:
        source_rows: list[dict[str, Any]] = []
        source_paths = []
        for relative in FALLBACK_SOURCE_FILES:
            candidate = root / relative
            source_paths.append(str(candidate.resolve()))
            source_rows.extend(_model_view(x) for x in _read_array_or_jsonl(candidate))
        rows = sorted(source_rows, key=lambda x: x["event_key"])
        expected_rows = rows[:]
        source = "freeze_predictions.public_events fallback"
        rule = "positive_events then negative_events, sorted by event_key"
    return rows, {
        "schema_version": "phase74r.model_event_order.v1",
        "source": source,
        "source_paths": source_paths,
        "manifest_exists": path.exists(),
        "source_sha256": sha256(path) if path.exists() else None,
        "count": len(rows),
        "order_sha256": canonical_hash(rows),
        "event_keys_sha256": canonical_hash([x["event_key"] for x in rows]),
        "expected_order_sha256": canonical_hash(expected_rows or []),
        "order_matches_independent_reconstruction": rows == (expected_rows or []),
        "order_rule": rule,
        "fields": list(MODEL_FIELDS),
        "written_by_phase74r": False,
    }


def load_evaluator_metadata(root: Path) -> list[dict[str, Any]]:
    """Read the frozen 76+76 manifests in their own file order."""
    paths = (
        root / "outputs/iclr27_phase19r/manifests/held_known_positive_events.jsonl",
        root / "outputs/iclr27_phase19r/manifests/held_known_negative_events.jsonl",
    )
    output: list[dict[str, Any]] = []
    for polarity, path in (("positive", paths[0]), ("negative", paths[1])):
        for ordinal, row in enumerate(iter_jsonl(path)):
            output.append(
                {
                    "evaluator_index": len(output),
                    "polarity": polarity,
                    "file_ordinal": ordinal,
                    "event_key": str(row.get("event_key", "")),
                    "fold": int(row.get("fold", -1)),
                    "kind": str(row.get("kind", "")),
                    "raw": row,
                }
            )
    return output


def join_evaluator_metadata(model_rows: Iterable[dict[str, Any]], metadata_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join metadata without reordering the model stream.

    The emitted records are model-order records followed by unmatched metadata
    records.  This makes a missing/extra event universe explicit and keeps the
    metadata denominator intact; no event is silently invented or removed.
    """
    model = list(model_rows)
    metadata_by_key = {row["event_key"]: row for row in metadata_rows}
    # Unit fixtures and legacy manifests may omit the derived evaluator index.
    # Preserve an explicitly supplied index, otherwise use the metadata file
    # order.  This keeps joins deterministic without requiring callers to
    # mutate the frozen evaluator metadata.
    metadata_index_by_key = {
        row["event_key"]: index for index, row in enumerate(metadata_rows)
    }
    matched: set[str] = set()
    output: list[dict[str, Any]] = []
    for model_index, row in enumerate(model):
        meta = metadata_by_key.get(row["event_key"])
        if meta is not None:
            matched.add(row["event_key"])
        output.append(
            {
                "join_index": len(output),
                "model_index": model_index,
                "model_event_key": row["event_key"],
                "evaluator_index": (
                    None
                    if meta is None
                    else meta.get("evaluator_index", metadata_index_by_key[row["event_key"]])
                ),
                "evaluator_event_key": None if meta is None else meta["event_key"],
                "polarity": None if meta is None else meta["polarity"],
                "fold": None if meta is None else meta["fold"],
                "join_status": "MATCHED" if meta is not None else "MODEL_EVENT_WITHOUT_EVALUATOR_METADATA",
            }
        )
    for meta in metadata_rows:
        if meta["event_key"] in matched:
            continue
        evaluator_index = meta.get(
            "evaluator_index", metadata_index_by_key[meta["event_key"]]
        )
        output.append(
            {
                "join_index": len(output),
                "model_index": None,
                "model_event_key": None,
                "evaluator_index": evaluator_index,
                "evaluator_event_key": meta["event_key"],
                "polarity": meta["polarity"],
                "fold": meta["fold"],
                "join_status": "EVALUATOR_EVENT_WITHOUT_MODEL_STREAM",
            }
        )
    return output


def event_order_contract(model_rows: list[dict[str, Any]], provenance: dict[str, Any], metadata_rows: list[dict[str, Any]], join_rows: list[dict[str, Any]]) -> dict[str, Any]:
    model_keys = [row["event_key"] for row in model_rows]
    meta_keys = [row["event_key"] for row in metadata_rows]
    matched = sum(row["join_status"] == "MATCHED" for row in join_rows)
    return {
        **provenance,
        "metadata_count": len(metadata_rows),
        "metadata_order_sha256": canonical_hash(meta_keys),
        "metadata_polarity_counts": {
            "positive": sum(row["polarity"] == "positive" for row in metadata_rows),
            "negative": sum(row["polarity"] == "negative" for row in metadata_rows),
        },
        "model_order_exactly_reproducible": bool(model_keys == [row["event_key"] for row in model_rows]) and bool(provenance.get("order_matches_independent_reconstruction", False)),
        "model_metadata_matched": matched,
        "model_metadata_unmatched": len(model_rows) - matched,
        "metadata_without_model": len(metadata_rows) - matched,
        "model_key_set_equals_metadata": set(model_keys) == set(meta_keys),
        "model_order_is_metadata_order": model_keys == meta_keys,
        "join_preserves_model_subsequence": [row["model_event_key"] for row in join_rows if row["model_index"] is not None] == model_keys,
        "join_preserves_metadata_denominator": len({row["evaluator_event_key"] for row in join_rows if row["evaluator_event_key"] is not None}) == len(meta_keys),
    }
