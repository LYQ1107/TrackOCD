#!/usr/bin/env python3
"""Run the pinned category-aware TETA reference for one native stream.

This helper is invoked with the dedicated environment that contains TETA.  It
is intentionally separate from the category-free TAO-OW evaluator: native
category IDs are used here only for the requested reference number and never
enter the TrackOCD backend.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


TETA_FIELDS = (
    "TETA",
    "LocS",
    "AssocS",
    "ClsS",
    "LocRe",
    "LocPr",
    "AssocRe",
    "AssocPr",
    "ClsRe",
    "ClsPr",
)


def _atomic_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp.%d" % (path.name, os.getpid()))
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _as_vector(value: Any) -> Optional[List[Optional[float]]]:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != len(TETA_FIELDS):
        return None
    result: List[Optional[float]] = []
    for item in value:
        try:
            number = float(item)
        except (TypeError, ValueError):
            return None
        result.append(number if math.isfinite(number) else None)
    return result


def _mean_vectors(values: Sequence[List[Optional[float]]]) -> Optional[List[Optional[float]]]:
    if not values:
        return None
    result: List[Optional[float]] = []
    for index in range(len(TETA_FIELDS)):
        finite = [float(row[index]) for row in values if row[index] is not None]
        result.append(float(np.mean(finite)) if finite else None)
    return result


def _class_vectors(
    combined: Dict[str, Any],
    category_names: Dict[int, str],
    known_ids: set[int],
    novel_ids: set[int],
) -> Dict[str, Optional[List[Optional[float]]]]:
    name_to_id = {str(name): int(category_id) for category_id, name in category_names.items()}
    groups: Dict[str, List[List[Optional[float]]]] = {"known": [], "novel": []}
    for key, cell in combined.items():
        if key == "average" or not isinstance(cell, dict):
            continue
        vector = _as_vector((cell.get("TETA") or {}).get(50))
        category_id = name_to_id.get(str(key))
        if vector is None or category_id is None:
            continue
        if category_id in known_ids:
            groups["known"].append(vector)
        elif category_id in novel_ids:
            groups["novel"].append(vector)
    return {name: _mean_vectors(values) for name, values in groups.items()}


def run(args: argparse.Namespace) -> Dict[str, Any]:
    # NumPy 2 removed aliases still referenced by the pinned TETA code.  Keep
    # compatibility local to this helper; no vendored source is modified.
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]
    try:
        import teta
        from teta.datasets import TAO
        from teta.eval import Evaluator
        from teta.metrics import TETA
    except Exception as exc:
        raise RuntimeError("TETA environment import failed") from exc

    tracker_json = Path(args.tracker_json).resolve()
    annotation = Path(args.annotation).resolve()
    output_root = Path(args.output_root).resolve()
    if not tracker_json.is_file():
        raise FileNotFoundError(tracker_json)
    if not annotation.is_file():
        raise FileNotFoundError(annotation)
    tracker_root = output_root / "trackers"
    tracker_data = tracker_root / args.tracker_name / "data"
    tracker_data.mkdir(parents=True, exist_ok=True)
    target = tracker_data / "tao_track.json"
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(tracker_json)

    eval_config = teta.config.get_default_eval_config()
    eval_config.update({
        "USE_PARALLEL": False,
        "NUM_PARALLEL_CORES": 1,
        "BREAK_ON_ERROR": True,
        "RETURN_ON_ERROR": False,
        "PRINT_RESULTS": False,
        "PRINT_ONLY_COMBINED": True,
        "PRINT_CONFIG": False,
        "TIME_PROGRESS": False,
        "DISPLAY_LESS_PROGRESS": True,
        "OUTPUT_SUMMARY": True,
        "OUTPUT_EMPTY_CLASSES": True,
        "OUTPUT_TEM_RAW_DATA": True,
        # The pinned TETA release indexes ``COMBINED_SEQ`` while formatting
        # its summary.  With False it stores only the combined dict in
        # ``all_res`` and then raises KeyError('COMBINED_SEQ').  Keeping the
        # per-sequence results makes that upstream formatter path valid; the
        # helper still reads the same combined, class-averaged result.
        "OUTPUT_PER_SEQ_RES": True,
    })
    dataset_config = teta.config.get_default_dataset_config()
    dataset_config.update({
        "GT_FOLDER": str(annotation),
        "TRACKERS_FOLDER": str(tracker_root),
        "TRACKERS_TO_EVAL": [args.tracker_name],
        "TRACKER_SUB_FOLDER": "data",
        "OUTPUT_FOLDER": str(output_root / "results"),
        "OUTPUT_SUB_FOLDER": "",
        "MAX_DETECTIONS": 300,
        "PRINT_CONFIG": False,
    })
    evaluator = Evaluator(eval_config)
    dataset = TAO(dataset_config)
    evaluator.evaluate([dataset], [TETA(exhaustive=False)])
    summary_path = output_root / "results" / args.tracker_name / "teta_summary_results.pth"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    raw = pickle.load(summary_path.open("rb"))
    combined = raw.get("COMBINED_SEQ", {})
    combined_vector = _as_vector((combined.get("average", {}).get("TETA") or {}).get(50))
    categories = json.loads(annotation.read_text(encoding="utf-8")).get("categories", [])
    category_names = {int(row["id"]): str(row["name"]) for row in categories}
    known_ids = {int(value) for value in json.loads(Path(args.known_ids).read_text(encoding="utf-8"))}
    novel_ids = {int(value) for value in json.loads(Path(args.novel_ids).read_text(encoding="utf-8"))}
    role_vectors = _class_vectors(combined, category_names, known_ids, novel_ids)
    result = {
        "schema_version": "trackocd.v2.teta_reference.v1",
        "status": "COMPLETE",
        "evaluator": "pinned TETA TAO",
        "tracker_name": args.tracker_name,
        "tracker_json": str(tracker_json),
        "annotation": str(annotation),
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": __import__("hashlib").sha256(summary_path.read_bytes()).hexdigest(),
        "config": {
            "parallel": False,
            "cores": 1,
            "max_detections_per_image": 300,
            "exhaustive": False,
            "threshold": 50,
        },
        "teta50": {
            "combined": dict(zip(TETA_FIELDS, combined_vector)) if combined_vector else None,
            "known": dict(zip(TETA_FIELDS, role_vectors["known"])) if role_vectors["known"] else None,
            "novel": dict(zip(TETA_FIELDS, role_vectors["novel"])) if role_vectors["novel"] else None,
        },
        "category_count": len(category_names),
        "known_ids_count": len(known_ids),
        "novel_ids_count": len(novel_ids),
        "test_semantic_accessed": False,
    }
    _atomic_json(Path(args.output_json), result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker-json", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--tracker-name", default="trackocd_native")
    parser.add_argument("--known-ids", required=True)
    parser.add_argument("--novel-ids", required=True)
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as exc:
        failure = {
            "schema_version": "trackocd.v2.teta_reference.v1",
            "status": "FAILED_TETA_REFERENCE",
            "error": "%s: %s" % (type(exc).__name__, exc),
            "test_semantic_accessed": False,
        }
        _atomic_json(Path(args.output_json), failure)
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
