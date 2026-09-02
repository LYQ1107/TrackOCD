#!/usr/bin/env python3
"""Post-hoc O (observability) audit for the frozen Q0 TRAIN replay.

The OVTR process runs without event labels.  This script joins the native
lineage sidecar to the frozen evaluator rows *after* prediction generation
and computes the registered event/prefix observability quantities.  Physical
track IDs are compared only through evaluator IoU and are never fed to a
model.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "outputs/iclr27_phase75b"
ARCHIVE = Path("/data2/usr_for_deadline/trackocd_phase75b")
NATIVE = ARCHIVE / "event_full_sequence_repair2/native_lineage.jsonl"
FRAMES = ARCHIVE / "event_full_sequence_repair2/native_lineage.frames.jsonl"
EVENT_POS = ROOT / "outputs/iclr27_phase19r/manifests/held_known_positive_events.jsonl"
EVENT_NEG = ROOT / "outputs/iclr27_phase19r/manifests/held_known_negative_events.jsonl"
JOIN = ROOT / "outputs/iclr27_phase74s/manifests/evaluator_join_v2.jsonl"
ROWS = ROOT / "outputs/iclr27_phase17r/csv/public_rows_corrected.csv"
PREFIXES = (1, 2, 4, 8, 16)
IOU_THRESHOLD = 0.5
EVENT_RE = re.compile(r"^v(?P<video>\d+):p(?P<track>\d+)$")


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(value, encoding="utf-8")
    os.replace(tmp, path)


def parse_json_box(value: str | None) -> list[float] | None:
    if not value:
        return None
    try:
        box = json.loads(value)
        if len(box) != 4:
            return None
        return [float(x) for x in box]
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def box_iou(left: list[float] | None, right: list[float] | None) -> float:
    if left is None or right is None:
        return 0.0
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    la = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    ra = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = la + ra - inter
    return inter / union if union > 0 else 0.0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_tracklet(key: str) -> tuple[int, int]:
    match = EVENT_RE.match(key)
    if match is None:
        raise ValueError(f"invalid tracklet key: {key}")
    return int(match.group("video")), int(match.group("track"))


def load_event_rows() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    events = read_jsonl(EVENT_POS) + read_jsonl(EVENT_NEG)
    joins = {str(row["event_key"]): row for row in read_jsonl(JOIN)}
    if len(events) != 152 or len(joins) != 152 or set(row["event_key"] for row in events) != set(joins):
        raise RuntimeError("frozen event/join protocol is not an exact 152-row contract")
    with ROWS.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    by_key = {str(row["row_key"]): row for row in csv_rows}
    by_track: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in csv_rows:
        by_track[(int(row["video_id"]), int(row["track_id"]))].append(row)
    for rows in by_track.values():
        rows.sort(key=lambda row: (int(row["event_rank"]), int(row["frame_id"]), int(row["image_id"])))
    prepared: list[dict[str, Any]] = []
    for event in events:
        source_keys = list(event.get("source_tracklet_keys", []))
        if len(source_keys) != 1:
            raise RuntimeError(f"expected one source tracklet for {event['event_key']}")
        source_video, source_track = parse_tracklet(source_keys[0])
        target_video, target_track = parse_tracklet(str(event["target_tracklet_key"]))
        target_rows = []
        for row_key in event.get("target_row_keys", []):
            if row_key not in by_key:
                raise RuntimeError(f"target row missing from frozen CSV: {row_key}")
            target_rows.append(by_key[row_key])
        source_rows = by_track.get((source_video, source_track), [])
        prepared.append({
            "event": event,
            "join": joins[event["event_key"]],
            "source_rows": source_rows,
            "target_rows": target_rows,
            "source_video": source_video,
            "source_track": source_track,
            "target_video": target_video,
            "target_track": target_track,
        })
    return prepared, by_key


def load_replay_lookup(prepared: Iterable[dict[str, Any]]) -> tuple[dict[tuple[int, int], list[dict[str, Any]]], set[tuple[int, int]]]:
    image_keys: set[tuple[int, int]] = set()
    for item in prepared:
        for row in item["source_rows"] + item["target_rows"]:
            image_keys.add((int(row["video_id"]), int(row["image_id"])))
    native: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    with NATIVE.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (int(row["video_id"]), int(row["image_id"]))
            if key in image_keys and row.get("bbox_xyxy") is not None:
                native[key].append(row)
    frame_keys: set[tuple[int, int]] = set()
    with FRAMES.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (int(row["video_id"]), int(row["image_id"]))
            if key in image_keys:
                frame_keys.add(key)
    return native, frame_keys


def evaluate_rows(rows: list[dict[str, Any]], native: dict[tuple[int, int], list[dict[str, Any]]], frame_keys: set[tuple[int, int]], prefix_rows: int | None = None) -> dict[str, Any]:
    if prefix_rows is not None:
        rows = rows[:prefix_rows]
    details = []
    best_track_sequence: list[int] = []
    for row in rows:
        key = (int(row["video_id"]), int(row["image_id"]))
        candidates = native.get(key, [])
        gt_box = parse_json_box(row.get("gt_bbox_xyxy"))
        scored = sorted(((box_iou(parse_json_box(json.dumps(candidate["bbox_xyxy"])), gt_box), candidate) for candidate in candidates), key=lambda x: (-x[0], int(x[1]["candidate_rank"])))
        best_iou = float(scored[0][0]) if scored else 0.0
        best = scored[0][1] if scored else None
        if best is not None:
            best_track_sequence.append(int(best["physical_track_id"]))
        event_iou = float(row.get("row_iou") or 0.0)
        temporal_iou = float(row.get("track_temporal_iou") or 0.0)
        assigned = int(float(row.get("assigned") or 0)) == 1
        event_reliable = assigned and event_iou >= IOU_THRESHOLD
        q0_reliable = best_iou >= IOU_THRESHOLD
        details.append({
            "row_key": row["row_key"],
            "video_id": int(row["video_id"]),
            "image_id": int(row["image_id"]),
            "frame_id": int(row["frame_id"]),
            "event_assigned": assigned,
            "event_transformed_iou": event_iou,
            "event_track_temporal_iou": temporal_iou,
            "q0_candidate_count": len(candidates),
            "q0_max_iou": best_iou,
            "q0_best_score": float(best["base_score"]) if best is not None and best.get("base_score") is not None else None,
            "q0_best_physical_track_id": int(best["physical_track_id"]) if best is not None else None,
            "q0_best_lifecycle": best.get("lifecycle") if best is not None else None,
            "frame_trace_present": key in frame_keys,
            "event_reliable": event_reliable,
            "q0_reliable": q0_reliable,
            "joint_reliable": event_reliable and q0_reliable,
            "failure_reason": (
                "asset_missing" if key not in frame_keys else
                "no_detection" if not candidates else
                "event_assignment_or_iou" if not event_reliable else
                "q0_iou_below_0.5" if not q0_reliable else
                "reliable"
            ),
        })
    transitions = sum(1 for left, right in zip(best_track_sequence, best_track_sequence[1:]) if left != right)
    ambiguity = 0
    for detail in details:
        candidates = native.get((detail["video_id"], detail["image_id"]), [])
        if detail["q0_candidate_count"] > 1:
            ambiguity += 1
    return {
        "rows_used": len(rows),
        "candidate_rows": sum(int(x["q0_candidate_count"] > 0) for x in details),
        "candidate_count": sum(int(x["q0_candidate_count"]) for x in details),
        "q0_reliable_rows": sum(bool(x["q0_reliable"]) for x in details),
        "event_reliable_rows": sum(bool(x["event_reliable"]) for x in details),
        "joint_reliable_rows": sum(bool(x["joint_reliable"]) for x in details),
        "max_iou": max((float(x["q0_max_iou"]) for x in details), default=0.0),
        "mean_max_iou": sum(float(x["q0_max_iou"]) for x in details) / len(details) if details else 0.0,
        "median_max_iou": sorted(float(x["q0_max_iou"]) for x in details)[len(details) // 2] if details else 0.0,
        "fragmentation_transitions": transitions,
        "ambiguity_rows": ambiguity,
        "no_detection_rows": sum(x["failure_reason"] == "no_detection" for x in details),
        "asset_missing_rows": sum(x["failure_reason"] == "asset_missing" for x in details),
        "details": details,
    }


def make_public_dir(tag: str, run_dir: Path) -> Path:
    public = OUT_ROOT / f"observability/{tag}"
    if public.exists() or os.path.lexists(str(public)):
        raise RuntimeError(f"observability output already exists: {public}")
    public.parent.mkdir(parents=True, exist_ok=True)
    tmp = public.with_name(f".{public.name}.{os.getpid()}.tmp")
    os.symlink(str(run_dir.resolve()), tmp)
    os.replace(tmp, public)
    return public


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="observability_repair2")
    parser.add_argument("--run-id", default=f"phase75b-o-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}")
    args = parser.parse_args()
    required = [NATIVE, FRAMES, EVENT_POS, EVENT_NEG, JOIN, ROWS]
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("Q0 lineage, frozen event manifests, or corrected event rows are missing")
    prepared, _ = load_event_rows()
    native, frame_keys = load_replay_lookup(prepared)
    run_dir = ARCHIVE / args.tag
    if run_dir.exists():
        raise RuntimeError(f"observability run already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    public = make_public_dir(args.tag, run_dir)
    event_records: list[dict[str, Any]] = []
    for item in prepared:
        event = item["event"]
        for prefix in PREFIXES:
            # Source is a completed prior-video support track; target is the
            # causal prefix under test.  This is the same source/target
            # chronology used by the frozen Phase19R evaluator.
            source_eval = evaluate_rows(item["source_rows"], native, frame_keys)
            target_eval = evaluate_rows(item["target_rows"], native, frame_keys, prefix_rows=prefix)
            source_reliable = source_eval["joint_reliable_rows"] > 0
            target_reliable = target_eval["joint_reliable_rows"] > 0
            record = {
                "event_key": event["event_key"],
                "model_event_uid": item["join"]["model_event_uid"],
                "kind": event["kind"],
                "polarity": "positive" if event["kind"] == "positive_existing" else "negative",
                "fold": int(event["fold"]),
                "category_denominator_only": event.get("category_gt_denominator_only", event.get("target_category_gt_denominator_only")),
                "source_video": int(event["source_video"]),
                "target_video": int(event["target_video"]),
                "source_tracklet_key": item["event"]["source_tracklet_keys"][0],
                "target_tracklet_key": item["event"]["target_tracklet_key"],
                "prefix": prefix,
                "source": {k: v for k, v in source_eval.items() if k != "details"},
                "target": {k: v for k, v in target_eval.items() if k != "details"},
                "source_reliable": source_reliable,
                "target_reliable": target_reliable,
                "both_reliable": source_reliable and target_reliable,
                "perfect_correspondence_ceiling": source_reliable and target_reliable,
                "failure_reason": (
                    "source_and_target_unreliable" if not source_reliable and not target_reliable else
                    "source_unreliable" if not source_reliable else
                    "target_unreliable" if not target_reliable else
                    "reliable_both_sides"
                ),
                "source_row_details": source_eval["details"],
                "target_row_details": target_eval["details"],
            }
            event_records.append(record)
    by_prefix: dict[str, Any] = {}
    by_fold: dict[str, Any] = {}
    for prefix in PREFIXES:
        subset = [x for x in event_records if x["prefix"] == prefix]
        positives = [x for x in subset if x["polarity"] == "positive"]
        by_prefix[str(prefix)] = {
            "events": len(subset),
            "positive_events": len(positives),
            "source_reliable": sum(x["source_reliable"] for x in positives),
            "target_reliable": sum(x["target_reliable"] for x in positives),
            "both_reliable": sum(x["both_reliable"] for x in positives),
            "negative_both_reliable": sum(x["both_reliable"] for x in subset if x["polarity"] == "negative"),
            "source_video_coverage": len({x["source_video"] for x in positives if x["source_reliable"]}),
            "target_video_coverage": len({x["target_video"] for x in positives if x["target_reliable"]}),
            "category_coverage": len({x["category_denominator_only"] for x in positives if x["both_reliable"]}),
            "failure_reasons": dict(Counter(x["failure_reason"] for x in positives)),
        }
    for fold in range(4):
        subset = [x for x in event_records if x["prefix"] == 16 and x["polarity"] == "positive" and x["fold"] == fold]
        by_fold[str(fold)] = {
            "positive_events": len(subset),
            "source_reliable": sum(x["source_reliable"] for x in subset),
            "target_reliable": sum(x["target_reliable"] for x in subset),
            "both_reliable": sum(x["both_reliable"] for x in subset),
            "source_videos": sorted({x["source_video"] for x in subset if x["source_reliable"]}),
            "target_videos": sorted({x["target_video"] for x in subset if x["target_reliable"]}),
            "categories": sorted({x["category_denominator_only"] for x in subset if x["both_reliable"]}),
        }
    p16 = by_prefix["16"]
    fold_both = [by_fold[str(f)]["both_reliable"] for f in range(4)]
    # This is a pre-registered diagnostic threshold anchored to the historical
    # 25/76 raw observable count; it is not a controller/retrieval metric.
    gate_pass = bool(p16["both_reliable"] >= 25 and sum(x > 0 for x in fold_both) >= 3)
    summary = {
        "phase": "Phase75B-O",
        "status": "O_GATE_PASS" if gate_pass else "O_GATE_FAIL",
        "run_id": args.run_id,
        "created_utc": now(),
        "prefixes": list(PREFIXES),
        "event_count": len(prepared),
        "positive_event_count": sum(x["event"]["kind"] == "positive_existing" for x in prepared),
        "negative_event_count": sum(x["event"]["kind"] == "negative_new" for x in prepared),
        "native_path": str(NATIVE),
        "native_sha256": sha256(NATIVE),
        "frame_trace_path": str(FRAMES),
        "frame_trace_sha256": sha256(FRAMES),
        "event_manifest_sha256": {"positive": sha256(EVENT_POS), "negative": sha256(EVENT_NEG), "join": sha256(JOIN), "corrected_rows": sha256(ROWS)},
        "native_candidate_image_keys": len(native),
        "replayed_event_image_keys": len(frame_keys),
        "by_prefix": by_prefix,
        "by_fold_prefix16": by_fold,
        "gate_contract": {"prefix16_positive_both_reliable_min": 25, "min_nonzero_folds": 3, "denominator": 76, "reliable_rule": "event assigned == 1 AND event transformed IoU >= 0.5 AND Q0 max IoU >= 0.5"},
        "gate_pass": gate_pass,
        "model_labels_joined_before_inference": False,
        "public_dev_q1_sealed_accessed": False,
        "future_rows_or_tracks": False,
        "physical_ids_used_as_model_input": False,
        "next_action": "freeze physical stream and begin one registered representation route" if gate_pass else "stop before representation; proposal/physical observability remains insufficient",
    }
    atomic_json(run_dir / "by_prefix.json", by_prefix)
    atomic_json(run_dir / "by_fold_prefix16.json", by_fold)
    atomic_json(run_dir / "summary.json", summary)
    atomic_text(run_dir / "event_observability.jsonl", "".join(json.dumps(x, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n" for x in event_records))
    csv_path = run_dir / "event_observability.csv"
    csv_tmp = csv_path.with_name(f".{csv_path.name}.{os.getpid()}.tmp")
    with csv_tmp.open("w", newline="", encoding="utf-8") as handle:
        fields = ["event_key", "model_event_uid", "polarity", "fold", "category_denominator_only", "source_video", "target_video", "prefix", "source_reliable", "target_reliable", "both_reliable", "source_candidate_rows", "target_candidate_rows", "source_q0_max_iou", "target_q0_max_iou", "failure_reason"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in event_records:
            writer.writerow({
                **{field: row.get(field) for field in fields if field in row},
                "source_candidate_rows": row["source"]["candidate_rows"],
                "target_candidate_rows": row["target"]["candidate_rows"],
                "source_q0_max_iou": row["source"]["max_iou"],
                "target_q0_max_iou": row["target"]["max_iou"],
            })
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(csv_tmp, csv_path)
    atomic_text(public / ".done", "complete\n")
    atomic_text(OUT_ROOT / f"completion/{args.tag}.done", "PASS_O_OBSERVABILITY\n")
    atomic_json(OUT_ROOT / "observability_status.json", {**summary, "public_output": str(public), "archive_output": str(run_dir)})
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
