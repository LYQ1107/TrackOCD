#!/usr/bin/env python3
"""Compare bounded anonymous-Train score-contract smoke streams.

This audit reads only normalized physical-stream JSONL rows and small run
receipts.  It does not open images, load GT, or consume semantic categories.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size, "sha256": _sha256(path)}


def _stream_stats(path: Path) -> dict[str, Any]:
    tracks = 0
    observations = 0
    videos: set[str] = set()
    lengths: list[int] = []
    score_count = 0
    invalid_scores = 0
    score_min = math.inf
    score_max = -math.inf
    score_sum = 0.0
    score_under_005 = 0
    score_over_05 = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            tracks += 1
            videos.add(str(row["video_id"]))
            length = len(row["frame_ids"])
            lengths.append(length)
            observations += length
            quality = row.get("quality", [])
            if not isinstance(quality, list):
                raise ValueError(f"quality is not a list at {path}:{line_number}")
            for raw_score in quality:
                try:
                    score = float(raw_score)
                except (TypeError, ValueError):
                    invalid_scores += 1
                    continue
                if not math.isfinite(score):
                    invalid_scores += 1
                    continue
                score_count += 1
                score_sum += score
                score_min = min(score_min, score)
                score_max = max(score_max, score)
                score_under_005 += score < 0.05
                score_over_05 += score > 0.5
    return {
        "stream": _metadata(path),
        "tracks": tracks,
        "observations": observations,
        "videos": len(videos),
        "score_count": score_count,
        "invalid_scores": invalid_scores,
        "score_min": None if not score_count else score_min,
        "score_mean": None if not score_count else score_sum / score_count,
        "score_max": None if not score_count else score_max,
        "score_under_0.05": score_under_005,
        "score_over_0.5": score_over_05,
        "track_length_median": None if not lengths else statistics.median(lengths),
        "singleton_tracks": sum(length == 1 for length in lengths),
        "tracks_over_16_observations": sum(length > 16 for length in lengths),
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-stream", type=Path, required=True)
    parser.add_argument("--candidate-stream", type=Path, required=True)
    parser.add_argument("--baseline-receipt", type=Path, required=True)
    parser.add_argument("--candidate-receipt", type=Path, required=True)
    parser.add_argument("--baseline-mechanism", default="tracked_double_softmax_max_class")
    parser.add_argument("--candidate-mechanism", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = _stream_stats(args.baseline_stream)
    candidate = _stream_stats(args.candidate_stream)
    baseline_receipt = _load_json(args.baseline_receipt)
    candidate_receipt = _load_json(args.candidate_receipt)
    result = {
        "schema_version": "trackocd.cvpr.score_contract_smoke_comparison.v1",
        "status": "COMPLETE_BOUNDED_ANONYMOUS_TRAIN_COMPARISON",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "contract": {
            "images_opened_by_audit": False,
            "gt_read_by_audit": False,
            "semantic_categories_consumed_by_audit": False,
            "formal_metrics": False,
            "training_started": False,
            "tao_val_accessed": False,
            "tao_test_accessed": False,
        },
        "baseline": {"stream": baseline, "receipt": _metadata(args.baseline_receipt)},
        "candidate": {"stream": candidate, "receipt": _metadata(args.candidate_receipt)},
        "mechanisms": {
            "baseline": args.baseline_mechanism,
            "candidate": args.candidate_mechanism,
        },
        "receipt_status": {
            "baseline": baseline_receipt.get("status"),
            "candidate": candidate_receipt.get("status"),
            "candidate_config": candidate_receipt.get("config"),
        },
        "native_output_identity": {
            "baseline_bbox_sha256": baseline_receipt.get("native_bbox_output_sha256"),
            "candidate_bbox_sha256": candidate_receipt.get("native_bbox_output_sha256"),
            "bbox_byte_identical": (
                baseline_receipt.get("native_bbox_output_sha256")
                == candidate_receipt.get("native_bbox_output_sha256")
            ),
            "baseline_track_sha256": baseline_receipt.get("native_track_output_sha256"),
            "candidate_track_sha256": candidate_receipt.get("native_track_output_sha256"),
            "track_byte_identical": (
                baseline_receipt.get("native_track_output_sha256")
                == candidate_receipt.get("native_track_output_sha256")
            ),
        },
        "deltas_candidate_minus_baseline": {
            "tracks": candidate["tracks"] - baseline["tracks"],
            "observations": candidate["observations"] - baseline["observations"],
            "score_mean": candidate["score_mean"] - baseline["score_mean"],
            "track_length_median": candidate["track_length_median"] - baseline["track_length_median"],
        },
        "interpretation": (
            f"The candidate is a bounded {args.candidate_mechanism} score-contract probe.  The paired run uses the "
            "same anonymous Train input, checkpoint, and frozen frontend; the changed score distribution "
            "and track counts prove that the selected shadow source path affects the actual exported route. "
            "This is mechanism evidence only and produces no OCD/MOT or Val qualification."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
