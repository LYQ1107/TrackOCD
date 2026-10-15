#!/usr/bin/env python3
"""Prepare the local TAO-Val manifest expected by public COVTrack.

The recovered public COVTrack parser uses a per-video ``frame_id`` field,
whereas the canonical local TAO annotation uses ``frame_index``.  This
helper creates a deterministic runtime-only copy with that structural field
and materializes the public LVIS class-name file from the pinned source.

This is an inference input preparation step, not model training, model
selection, or an OCD evaluator.  The derived files live below the ignored
``outputs/trackocd_v2_runtime`` tree and are never uploaded.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANNOTATION = ROOT / "data/raw/tao/annotations/validation.json"
DEFAULT_SOURCE = ROOT / "vendor/trackocd_external/covtrack_public"
DEFAULT_OUTPUT = ROOT / "outputs/trackocd_v2_runtime/frontend_inputs/covtrack_val"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.tmp.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_lvis_classes(source_root: Path) -> tuple[Path, list[str]]:
    source_file = source_root / "ovtrack/models/roi_heads/class_name.py"
    tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
    for node in tree.body:
        targets = [target.id for target in node.targets if isinstance(target, ast.Name)] if isinstance(node, ast.Assign) else []
        if "LVIS_CLASSES" in targets:
            value = ast.literal_eval(node.value)
            classes = [str(item) for item in value]
            if len(classes) != 1203 or not all(classes):
                raise ValueError(f"pinned LVIS_CLASSES is not the expected 1203-name list: {len(classes)}")
            return source_file, classes
    raise ValueError(f"LVIS_CLASSES assignment not found in {source_file}")


def _derive_annotation(annotation: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    images = annotation.get("images")
    if not isinstance(images, list):
        raise ValueError("TAO annotation has no images list")
    by_video: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for image in images:
        if not isinstance(image, dict):
            raise ValueError("TAO image entry is not an object")
        if "video_id" not in image or "frame_index" not in image:
            raise ValueError("TAO image lacks video_id or frame_index")
        by_video[int(image["video_id"])].append(image)

    frame_counts: dict[str, int] = {}
    updated_by_id: dict[int, dict[str, Any]] = {}
    for video_id, rows in by_video.items():
        rows.sort(key=lambda row: (int(row["frame_index"]), int(row["id"])))
        for frame_id, image in enumerate(rows):
            updated = dict(image)
            updated["frame_id"] = frame_id
            updated_by_id[int(updated["id"])] = updated
        frame_counts[str(video_id)] = len(rows)

    derived = dict(annotation)
    derived["images"] = [updated_by_id[int(image["id"])] for image in images]
    return derived, frame_counts


def prepare(
    *,
    annotation_path: Path = DEFAULT_ANNOTATION,
    source_root: Path = DEFAULT_SOURCE,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    annotation_path = annotation_path.resolve()
    source_root = source_root.resolve()
    output_dir = output_dir.resolve()
    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)

    source_file, classes = _load_lvis_classes(source_root)
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    derived, frame_counts = _derive_annotation(annotation)
    output_dir.mkdir(parents=True, exist_ok=True)
    derived_path = output_dir / "validation_ours_v1.json"
    classes_path = output_dir / "lvis_classes_v1.txt"
    report_path = output_dir / "preparation.json"
    _atomic_write(derived_path, json.dumps(derived, ensure_ascii=False, separators=(",", ":")) + "\n")
    _atomic_write(classes_path, "\n".join(classes) + "\n")
    report = {
        "schema_version": "trackocd.v2.covtrack_val_input_preparation.v1",
        "status": "COMPLETE",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input": {
            "annotation": str(annotation_path),
            "bytes": annotation_path.stat().st_size,
            "sha256": _sha256(annotation_path),
        },
        "source": {
            "root": str(source_root),
            "class_source": str(source_file),
            "class_source_sha256": _sha256(source_file),
            "class_count": len(classes),
        },
        "outputs": {
            "annotation": str(derived_path),
            "annotation_bytes": derived_path.stat().st_size,
            "annotation_sha256": _sha256(derived_path),
            "classes": str(classes_path),
            "classes_bytes": classes_path.stat().st_size,
            "classes_sha256": _sha256(classes_path),
            "report": str(report_path),
        },
        "counts": {
            "videos": len(derived.get("videos", [])),
            "images": len(derived.get("images", [])),
            "annotations": len(derived.get("annotations", [])),
            "tracks": len(derived.get("tracks", [])),
            "categories_in_annotation": len(derived.get("categories", [])),
            "frame_counts_by_video": frame_counts,
        },
        "boundary": {
            "split": "tao_val",
            "tao_test_read": False,
            "training_started": False,
            "model_selection": False,
            "novel_vocabulary_used_for_selection": False,
            "role": "runtime_dataset_manifest_and_fixed_public_frontend_class_source_only",
        },
    }
    _atomic_write(report_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=Path, default=DEFAULT_ANNOTATION)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(prepare(annotation_path=args.annotation, source_root=args.source_root, output_dir=args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
