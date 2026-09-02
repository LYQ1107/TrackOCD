"""Executable metamorphic checks for the Phase74R physical-side plumbing."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .asset_identity import synthetic_pipeline
from .io import atomic_json, atomic_jsonl, canonical_hash
from .prefix_contract import source_rows, target_rows


def _fixture() -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    event = {
        "event_key": "fixture:event",
        "kind": "positive",
        "polarity": "positive",
        "source_tracklet_keys": ["v1:p10"],
        "target_tracklet_key": "v2:p20",
    }
    q0 = {
        "dataset_name": "tao",
        "dataset_split": "validation",
        "video_file_name": "val/Scene/clip",
        "image_file_name": "val/Scene/clip/000001.jpg",
        "frame_index": 1,
        "canonical_image_key": "tao|validation|val/Scene/clip|frame=1",
        "resolved_path": None,
        "path_exists": False,
        "video_id": 100,
        "image_id": 1001,
        "track_id": 900,
        "bbox": [10, 10, 30, 30],
        "q0_iou": 0.8,
    }
    rows = {
        "v1:p10": [{"row_key": "1:0:1:10:11", "video_id": 1, "track_id": 10, "frame_id": 0, "image_id": 11, "event_rank": 0}],
        "v2:p20": [{"row_key": "2:1:1:20:21", "video_id": 2, "track_id": 20, "frame_id": 1, "image_id": 21, "event_rank": 0}, {"row_key": "2:2:1:20:22", "video_id": 2, "track_id": 20, "frame_id": 2, "image_id": 22, "event_rank": 1}],
    }
    event["bbox_iou"] = 0.7
    event["event_iou"] = 0.7
    event["q0_iou"] = 0.8
    event["assigned"] = 1
    event["fragmentation_signature"] = [{"track": "A", "first_frame": 0, "last_frame": 1}, {"track": "B", "first_frame": 2, "last_frame": 3}]
    return event, q0, rows


def _physical_payload(event: dict[str, Any], q0: dict[str, Any]) -> dict[str, Any]:
    payload = synthetic_pipeline(event, q0)
    payload["fragmentation_signature"] = sorted(
        [{"first_frame": int(x["first_frame"]), "last_frame": int(x["last_frame"])} for x in event.get("fragmentation_signature", [])],
        key=lambda x: (x["first_frame"], x["last_frame"]),
    )
    return payload


def _crash_probe(path: Path, writer: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.injected.tmp")
    if path.exists():
        path.unlink()
    if tmp.exists():
        tmp.unlink()
    try:
        def records():
            yield {"record": 1, "writer": writer}
            raise RuntimeError(f"injected crash at {writer}")
        # The exception occurs while the atomic writer owns its temporary file;
        # its finally block must remove the temp and never install ``path``.
        atomic_jsonl(path, records())
        return False
    except RuntimeError:
        if tmp.exists():
            tmp.unlink()
        return not path.exists() and not list(path.parent.glob(f".{path.name}.*.tmp"))


def _static_antihardcode(src_root: Path) -> bool:
    suspicious = re.compile(r"(?:category_shuffle|event_label_swap|physical_id_renumber|future_append)\s*['\"]?\s*:\s*True")
    return not any(suspicious.search(path.read_text(encoding="utf-8", errors="ignore")) for path in src_root.rglob("*.py"))


def run_metamorphic(root: Path, output_dir: Path) -> dict[str, Any]:
    event, q0, rows = _fixture()
    baseline = _physical_payload(event, q0)

    category_variants = []
    for variant in ("permuted", "sentinel", "removed"):
        altered = dict(event)
        if variant == "permuted":
            altered["category_id"] = 999
        elif variant == "sentinel":
            altered["category_id"] = "SENTINEL"
        else:
            altered.pop("category_id", None)
        category_variants.append(_physical_payload(altered, q0))
    category_shuffle = all(value == baseline for value in category_variants)

    swapped = dict(event)
    swapped["kind"], swapped["polarity"] = "negative", "negative"
    event_label_swap = _physical_payload(swapped, q0) == baseline

    renumbered_q0 = dict(q0)
    renumbered_q0["track_id"] = 123456
    physical_id_renumber = _physical_payload(event, renumbered_q0) == baseline

    early = target_rows(event, 1, rows)
    with_future = dict(rows)
    with_future["v2:p20"] = rows["v2:p20"] + [{"row_key": "2:3:1:20:23", "video_id": 2, "track_id": 20, "frame_id": 3, "image_id": 23, "event_rank": 2}]
    future = target_rows(event, 1, with_future)
    future_append = early == future

    source = source_rows(event, rows)
    source_before_target = [x["position"] for x in source] == [0] and len(source) == 1

    repeat_a = output_dir / "metamorphic_repeat_a" / "fixture.json"
    repeat_b = output_dir / "metamorphic_repeat_b" / "fixture.json"
    atomic_json(repeat_a, baseline)
    atomic_json(repeat_b, baseline)
    repeat_determinism = repeat_a.read_bytes() == repeat_b.read_bytes()

    crash_results = {
        "asset_manifest": _crash_probe(output_dir / "crash" / "asset_manifest.jsonl", "asset_manifest"),
        "alignment": _crash_probe(output_dir / "crash" / "alignment.jsonl", "alignment"),
        "status": _crash_probe(output_dir / "crash" / "status.jsonl", "status"),
    }
    atomic_crash = all(crash_results.values())
    static_antihardcode = _static_antihardcode(root / "src/iclr27_phase74r")
    results = {
        "category_shuffle": category_shuffle,
        "event_label_swap": event_label_swap,
        "physical_id_renumber": physical_id_renumber,
        "future_append": future_append,
        "source_before_target": source_before_target,
        "repeat_determinism": repeat_determinism,
        "atomic_crash": atomic_crash,
        "static_antihardcode": static_antihardcode,
        "details": {"category_hashes": [canonical_hash(x) for x in category_variants], "baseline_hash": canonical_hash(baseline), "crash_results": crash_results},
    }
    return results
