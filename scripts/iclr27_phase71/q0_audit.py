#!/usr/bin/env python3
"""Phase71 read-only Q0 lineage and score-contract audit.

The Q0 prediction stream is immutable evidence.  This script never writes to
an older phase namespace and never passes annotations to a model.  It
recomputes the class-agnostic recall diagnostic, checks the five-field CSV
lineage where available, and emits an explicit score-channel sidecar: Q0 was
run in ``score_mode=base`` so DSCT/objectness channels are *not produced*.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import pathlib
import statistics
import tempfile
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[2]
Q0_JSON = ROOT / "outputs/iclr27_phase4q/q0_long/teta_results/tao_track.json"
Q0_CSV = ROOT / "outputs/iclr27_phase4q/q0_long/proposals_dev.csv"
Q0_CKPT = ROOT / "outputs/iclr27_phase4q/q0_long/checkpoint.pth"
GT_JSON = ROOT / "data/external_annotations/ovtr/validation_ours_v1.json"
PHASE68_AUDIT = ROOT / "outputs/iclr27_phase68/audit/full_sequence_baseline.json"
OUT = ROOT / "outputs/iclr27_phase71/audit"


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def xywh_to_xyxy(box: Sequence[float]) -> Tuple[float, float, float, float]:
    x, y, w, h = map(float, box[:4])
    return x, y, x + max(w, 0.0), y + max(h, 0.0)


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(b)
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ab = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    den = aa + ab - inter
    return inter / den if den > 0 else 0.0


def recall_curve(gt_path: pathlib.Path, pred_path: pathlib.Path) -> Dict[str, Any]:
    # This is post-hoc scoring only; the annotations are never read by the
    # model/training path.  Keep the exact Phase68 denominator.
    gt_doc = json.loads(gt_path.read_text())
    pred_doc = json.loads(pred_path.read_text())
    gt: Dict[int, List[dict]] = defaultdict(list)
    for ann in gt_doc.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        gt[int(ann["image_id"])].append(ann)
    pred: Dict[int, List[dict]] = defaultdict(list)
    for row in pred_doc:
        if "image_id" in row and "bbox" in row:
            pred[int(row["image_id"])].append(row)
    totals = sum(len(v) for v in gt.values())
    out: Dict[str, Any] = {"gt_rows": totals, "topk": {}}
    for k in (1, 5, 20, 100, 0):
        hits = {0.3: 0, 0.5: 0, 0.7: 0}
        best_values: List[float] = []
        for image_id, anns in gt.items():
            ps = sorted(pred.get(image_id, []), key=lambda z: float(z.get("score", 0.0)), reverse=True)
            if k > 0:
                ps = ps[:k]
            for ann in anns:
                best = max((iou(ann["bbox"], p["bbox"]) for p in ps), default=0.0)
                best_values.append(best)
                for t in hits:
                    if best >= t:
                        hits[t] += 1
        out["topk"][str(k)] = {
            "thresholds": {f"{t:.1f}": {"matched_rows": n, "recall": n / max(totals, 1)} for t, n in hits.items()},
            "mean_best_iou": statistics.fmean(best_values) if best_values else 0.0,
            "median_best_iou": statistics.median(best_values) if best_values else 0.0,
        }
    out["prediction_count"] = len(pred_doc)
    out["prediction_images"] = len(pred)
    return out


def csv_audit(path: pathlib.Path) -> Dict[str, Any]:
    required = ["video_id", "frame_id", "image_id", "proposal_local_id", "track_id"]
    rows = 0
    key_set = set()
    order_digest = hashlib.sha256()
    fieldnames: List[str] = []
    malformed = 0
    score_stats: List[float] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            rows += 1
            try:
                key = tuple(int(row[k]) for k in required)
                key_set.add(key)
                order_digest.update(("|".join(str(x) for x in key) + "\n").encode())
                score_stats.append(float(row["score"]))
            except (KeyError, TypeError, ValueError):
                malformed += 1
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "rows": rows,
        "fieldnames": fieldnames,
        "required_five_field_key": required,
        "unique_key_count": len(key_set),
        "duplicate_key_count": rows - len(key_set),
        "malformed_rows": malformed,
        "ordered_key_digest": order_digest.hexdigest(),
        "score_min": min(score_stats) if score_stats else None,
        "score_max": max(score_stats) if score_stats else None,
        "score_mean": statistics.fmean(score_stats) if score_stats else None,
        "gt_fields_present_but_not_model_inputs": ["gt_role", "gt_iou", "gt_category_id"],
    }


def export_score_sidecar(path: pathlib.Path, out_path: pathlib.Path) -> Dict[str, Any]:
    """Export all immutable Q0 rows with explicit score-channel provenance.

    Missing channels are represented as null, rather than silently reusing
    base scores.  Compression avoids a second large uncompressed prediction
    file while preserving one-record-per-Q0-row auditability.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    sample: List[dict] = []
    with path.open() as src, gzip.open(out_path, "wt", encoding="utf-8") as dst:
        rows = json.load(src)
        for r in rows:
            rec = {
                "image_id": r.get("image_id"),
                "video_id": r.get("video_id"),
                "frame_id": r.get("frame_id"),
                "proposal_local_id": r.get("proposal_local_id"),
                "track_id": r.get("track_id"),
                "bbox": r.get("bbox"),
                "base_score": r.get("score"),
                "raw_score": r.get("score"),
                "pre_filter_score": None,
                "dsct_score": None,
                "objectness_score": None,
                "score_mode": "base",
                "missing_channel_reason": "Q0 score_mode=base export has no DSCT/objectness tensor or pre-filter score",
            }
            dst.write(json.dumps(rec, separators=(",", ":")) + "\n")
            count += 1
            if len(sample) < 5:
                sample.append(rec)
    return {"path": str(out_path.resolve()), "bytes": out_path.stat().st_size, "sha256": sha256(out_path), "records": count, "sample": sample}


def main() -> None:
    for p in (Q0_JSON, Q0_CSV, Q0_CKPT, GT_JSON, PHASE68_AUDIT):
        if not p.exists():
            raise FileNotFoundError(p)
    OUT.mkdir(parents=True, exist_ok=True)
    ref = json.loads(PHASE68_AUDIT.read_text())
    pred = json.loads(Q0_JSON.read_text())
    required = {"image_id", "bbox", "score", "category_id", "video_id", "track_id"}
    malformed = []
    scores = []
    for i, r in enumerate(pred[:10000]):
        if not required.issubset(r):
            malformed.append(i)
        try:
            scores.append(float(r["score"]))
            b = [float(x) for x in r["bbox"]]
            if len(b) != 4 or not all(math.isfinite(x) for x in b):
                malformed.append(i)
        except Exception:
            malformed.append(i)
    rec = recall_curve(GT_JSON, Q0_JSON)
    csv_info = csv_audit(Q0_CSV)
    side = export_score_sidecar(Q0_JSON, OUT / "q0_score_channels.jsonl.gz")
    obj = {
        "protocol": "trackocd_phase71_q0_read_only_score_mode_base_equivalence",
        "project_root": str(ROOT),
        "q0_checkpoint": {"path": str(Q0_CKPT.resolve()), "bytes": Q0_CKPT.stat().st_size, "sha256": sha256(Q0_CKPT)},
        "q0_prediction": {"path": str(Q0_JSON.resolve()), "bytes": Q0_JSON.stat().st_size, "sha256": sha256(Q0_JSON), "records": len(pred), "sample_malformed_indices": malformed[:20], "sample_score_min": min(scores), "sample_score_max": max(scores)},
        "q0_reference_phase68": {"path": str(PHASE68_AUDIT.resolve()), "sha256": sha256(PHASE68_AUDIT), "prediction_count": ref.get("prediction_count"), "reference_top20_iou05": ref.get("recall", {}).get("topk", {}).get("20", {}).get("thresholds", {}).get("0.5", {}).get("recall")},
        "recomputed_recall": rec,
        "csv_lineage": csv_info,
        "score_channels": side,
        "score_contract": {"score_mode": "base", "base_score_source": "track_instances.pred_logits.sigmoid().max", "raw_score_equals_base": True, "pre_filter_score": "not exported by historical Q0 JSON", "dsct_score": "not produced (DSCT disabled)", "objectness_score": "not produced (class-agnostic adapter absent)", "physical_assignment": "Q0 output track_id retained only as bookkeeping; no semantic input", "row_key": "(video_id, frame_id, image_id, proposal_local_id, track_id) where CSV lineage exposes all five"},
        "equivalence_checks": {"prediction_count_match": len(pred) == ref.get("prediction_count"), "prediction_sha_match": sha256(Q0_JSON) == ref.get("q0_json", {}).get("sha256"), "recall_top20_iou05_match": abs(rec["topk"]["20"]["thresholds"]["0.5"]["recall"] - ref["recall"]["topk"]["20"]["thresholds"]["0.5"]["recall"]) < 1e-12, "csv_duplicate_keys": csv_info["duplicate_key_count"], "csv_malformed_rows": csv_info["malformed_rows"]},
        "forbidden_inputs_not_read_for_training": ["DEV+", "Q1", "public new-model labels", "future frames/tracks", "held GT as model input", "semantic/physical IDs as features", "category text"],
        "labels_used_for_model": False,
    }
    atomic_json(OUT / "q0_equivalence.json", obj)
    atomic_json(OUT / "q0_score_channels_summary.json", side)
    (OUT.parent / "completion").mkdir(parents=True, exist_ok=True)
    (OUT.parent / "completion/q0_audit.done").write_text("complete\n")
    print(json.dumps({"q0_records": len(pred), "top20_iou05": rec["topk"]["20"]["thresholds"]["0.5"]["recall"], "prediction_sha": obj["q0_prediction"]["sha256"], "sidecar": side["path"]}, indent=2))


if __name__ == "__main__":
    main()
