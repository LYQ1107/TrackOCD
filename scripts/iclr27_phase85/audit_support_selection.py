#!/usr/bin/env python3
"""Post-hoc event taxonomy for the frozen Phase85 support selection.

This audit never changes a choice or trains a model.  It joins each selected
candidate back to its native box and reports the candidate-pool upper bound,
raw/reranked choices and defer decisions on the fixed 76+76 event set.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import sys
if str(ROOT := Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(ROOT))

NATIVE = Path("/data2/usr_for_deadline/trackocd_phase83/a2_full/native_lineage.jsonl")
PUBLIC = ROOT / "data/iclr27_phase19r/sources/public_rows_corrected.csv"
REPLAY = ROOT / "outputs/iclr27_phase85/metrics/support_event_replay.json"
OUT = ROOT / "outputs/iclr27_phase85"
FEATURES = Path("/data2/usr_for_deadline/trackocd_phase83/a2_dino_full_r1/merged/native_dinov2.npz")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def box(v: Any) -> list[float] | None:
    try:
        x = [float(z) for z in (json.loads(v) if isinstance(v, str) else v)]
        return x if len(x) == 4 else None
    except Exception:
        return None


def iou(a: list[float] | None, b: list[float] | None) -> float:
    if a is None or b is None:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-8)


def track_key(r: dict[str, str]) -> str:
    return f"v{int(r['video_id'])}:p{int(r['track_id'])}"


def order_native(r: dict[str, Any], idx: int) -> tuple[int, int, int]:
    return (int(r.get("candidate_rank") or 0), int(r.get("proposal_local_id") or 0), idx)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def main() -> None:
    native = [json.loads(line) for line in NATIVE.open(encoding="utf-8") if line.strip()]
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    native_by_identity: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for i, row in enumerate(native):
        key = (int(row.get("video_id", -1)), int(row.get("image_id", -1)))
        groups[key].append(i)
        native_by_identity[(key[0], key[1], int(row.get("proposal_local_id") or -1))].append(i)
    for key in groups:
        groups[key].sort(key=lambda i: order_native(native[i], i))
    public = list(csv.DictReader(PUBLIC.open(newline="", encoding="utf-8")))
    public_by_row = {str(r.get("row_key")): r for r in public}
    public_by_identity: dict[tuple[int, int, int], list[dict[str, str]]] = defaultdict(list)
    track_categories: dict[str, int] = {}
    for row in public:
        ident = (int(row["video_id"]), int(row["image_id"]), int(row.get("proposal_local_id") or -1))
        public_by_identity[ident].append(row)
        try:
            track_categories[track_key(row)] = int(float(row.get("gt_category_id_common", -1) or -1))
        except Exception:
            track_categories[track_key(row)] = -1
    replay = json.loads(REPLAY.read_text(encoding="utf-8"))
    features = np.asarray(np.load(FEATURES, allow_pickle=False)["features"], np.float32)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    source_path = Path(replay["inputs"]["source_cache"])
    source_z = np.load(source_path, allow_pickle=False)
    source_keys = [str(x) for x in source_z["keys"].tolist()]
    source_index = {key: i for i, key in enumerate(source_keys)}
    source_vectors = np.asarray(source_z["vectors"], np.float32)
    from src.iclr27_phase85.raw_candidate_anchor import stable_raw_topk
    event_records = replay["records"]
    rows: list[dict[str, Any]] = []
    for event in event_records:
        source_cat = track_categories.get(str(event.get("source_tracklet_key")), -1)
        for target in event.get("target_rows", []):
            gt_row = public_by_row.get(str(target.get("row_key")))
            gt_box = box(gt_row.get("gt_bbox_xyxy")) if gt_row else None
            target_key = (int(target.get("video_id", -1)), int(target.get("image_id", -1)))
            cand_ids = groups.get(target_key, [])
            # The replay's *_choice_rank is an index in the stable raw top-K
            # array, not the original native candidate rank.
            source_key = str(event.get("source_tracklet_key"))
            if cand_ids and source_key in source_index:
                src = source_vectors[4, source_index[source_key]]
                scores = features[np.asarray(cand_ids, np.int64)] @ src
                order = stable_raw_topk(scores, int(target.get("topk_count", 0)))
                topk = [cand_ids[int(j)] for j in order]
            else:
                topk = cand_ids[: int(target.get("topk_count", 0))]
            pool_ious = [iou(box(native[i].get("bbox_xyxy")), gt_box) for i in topk]
            pool_max = max(pool_ious, default=0.0)
            raw_iou = float(target.get("raw_iou", 0.0)); rank_iou = float(target.get("reranked_iou", 0.0))
            defer = bool(target.get("defer", False))
            if not cand_ids:
                bucket = "no_candidate_frame"
            elif pool_max < 0.5:
                bucket = "pool_no_iou_ge_0.5"
            elif defer and rank_iou >= 0.5:
                bucket = "defer_with_reliable_rerank"
            elif defer:
                bucket = "defer_with_unreliable_rerank"
            elif rank_iou >= 0.5 and raw_iou < 0.5:
                bucket = "rerank_rescue"
            elif rank_iou < 0.5 and raw_iou >= 0.5:
                bucket = "rerank_harm"
            elif rank_iou >= 0.5:
                bucket = "both_reliable"
            else:
                bucket = "pool_has_candidate_not_selected"
            rows.append({
                "event_key": str(event.get("event_key")), "model_event_uid": str(event.get("model_event_uid")),
                "fold": int(event.get("fold", -1)), "polarity": str(event.get("polarity")), "prefix": int(event.get("prefix", 0)),
                "source_tracklet_key": str(event.get("source_tracklet_key")), "source_category": source_cat,
                "target_row_key": str(target.get("row_key")), "candidate_count": len(cand_ids),
                "topk_count": len(topk), "pool_max_iou": float(pool_max), "raw_iou": raw_iou,
                "reranked_iou": rank_iou, "defer": defer, "bucket": bucket,
            })
    summary = []
    for prefix in (1, 2, 4, 8, 16):
        for polarity in ("positive", "negative"):
            subset = [r for r in rows if r["prefix"] == prefix and r["polarity"] == polarity]
            counts = Counter(r["bucket"] for r in subset)
            summary.append({"prefix": prefix, "polarity": polarity, "rows": len(subset), "buckets": dict(sorted(counts.items())), "pool_reliable_rows": sum(r["pool_max_iou"] >= 0.5 for r in subset), "raw_reliable_rows": sum(r["raw_iou"] >= 0.5 for r in subset), "reranked_reliable_rows": sum(r["reranked_iou"] >= 0.5 for r in subset), "deferred_rows": sum(r["defer"] for r in subset)})
    result = {
        "schema_version": "trackocd.phase85.support_selection_audit.v1", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "replay": str(REPLAY.resolve()), "replay_sha256": sha(REPLAY), "native": str(NATIVE.resolve()), "native_sha256": sha(NATIVE),
        "rows": rows, "summary": summary, "denominator": {"positive_events": 76, "negative_events": 76, "prefixes": [1, 2, 4, 8, 16]},
        "note": "Pool ceiling uses the first fixed candidate_count ordering as a conservative audit; raw/reranked selected IoUs are copied from the frozen replay. Labels are post-hoc only.",
        "public_dev_q1_sealed_accessed": False, "future_rows_or_tracks": False, "ids_as_model_input": False,
    }
    atomic_json(OUT / "audit/support_selection_audit.json", result)
    atomic_json(OUT / "completion/support_selection_audit.done", {"status": "DONE", "audit": str((OUT / "audit/support_selection_audit.json").resolve()), "sha256": sha(OUT / "audit/support_selection_audit.json")})
    print(json.dumps({"p16": [s for s in summary if s["prefix"] == 16]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
