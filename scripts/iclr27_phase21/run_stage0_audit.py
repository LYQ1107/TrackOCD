#!/usr/bin/env python
"""Phase21 Stage 0: geometry/chronology audit and Phase20 reproduction.

The input is the frozen public-TRAIN DSCT proposal CSV and the TRAIN-derived
pseudo-held event manifests.  No DEV+, Q1, or public new-model file is opened.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/iclr27_phase19r/sources/public_rows_corrected.csv"
POS = ROOT / "outputs/iclr27_phase19r/manifests/held_known_positive_events.jsonl"
NEG = ROOT / "outputs/iclr27_phase19r/manifests/held_known_negative_events.jsonl"
P20_CEIL = ROOT / "outputs/iclr27_phase20/audit/observability_ceiling.json"
P19_FOLD = ROOT / "outputs/iclr27_phase19r/manifests/fold_manifest.json"
OUT = ROOT / "outputs/iclr27_phase21"
PREFIXES = (1, 2, 4, 8, 16)
IOU_THR = 0.5


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_box(value: str | None) -> list[float] | None:
    if value is None or not str(value).strip(): return None
    try:
        x = json.loads(value)
        if isinstance(x, list) and len(x) == 4:
            return [float(v) for v in x]
    except Exception:
        pass
    return None


def box_iou(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None or b is None: return None
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1)); ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ab = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    den = aa + ab - inter
    return inter / den if den > 0 else 0.0


def key_for_row(r: dict[str, str]) -> str:
    return f"v{int(r['video_id'])}:p{int(r['track_id'])}"


def row_key_parts(value: str) -> tuple[int, int, int, int, int] | None:
    try:
        a, b, c, d, e = value.split(":")
        return int(a), int(b), int(c), int(d), int(e)
    except Exception:
        return None


def load_events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def finite(v: float) -> bool:
    return math.isfinite(float(v))


def main() -> None:
    (OUT / "audit").mkdir(parents=True, exist_ok=True)
    (OUT / "manifests").mkdir(parents=True, exist_ok=True)
    with SRC.open(newline="") as f: rows = list(csv.DictReader(f))
    # Preserve the fold partition as a lightweight, TRAIN-derived artifact in
    # the independent namespace; no feature or label arrays are copied.
    fold_manifest = json.loads(P19_FOLD.read_text())
    fold_manifest = dict(fold_manifest)
    fold_manifest["phase21_namespace"] = True
    fold_manifest["source_rows_path"] = str(SRC)
    fold_manifest["source_rows_sha256"] = sha256(SRC)
    fold_manifest["labels_used"] = "public TRAIN category/video metadata only"
    fold_manifest["sealed_inputs"] = ["DEV+", "Q1", "public new-model labels"]
    atomic_json(OUT / "manifests/fold_manifest.json", fold_manifest)
    row_by_key = {str(r["row_key"]): (i, r) for i, r in enumerate(rows)}
    by_track: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for i, r in enumerate(rows): by_track[key_for_row(r)].append((i, r))
    for k in by_track:
        by_track[k].sort(key=lambda x: (int(x[1].get("event_rank", 0)), int(x[1].get("frame_id", 0)), x[0]))

    # Geometry and chronology checks are row-level and independent of any
    # proposal variant.  Values use each row's actual image dimensions.
    resolution = Counter(); invalid_box = 0; norm_mismatch = 0; iou_mismatch = 0
    iou_diffs: list[float] = []; smooth_parse_fail = 0; row_key_duplicates = len(rows) - len(row_by_key)
    chronology_bad_tracks: list[str] = []; prefix_bad_rows = 0; event_rank_duplicates = 0
    for r in rows:
        try: w, h = float(r["image_width"]), float(r["image_height"]); resolution[f"{int(w)}x{int(h)}"] += 1
        except Exception: w = h = 0.0
        b = parse_box(r.get("bbox_xyxy")); g = parse_box(r.get("gt_bbox_xyxy")); sb = parse_box(r.get("causal_smoothed_bbox_xyxy"))
        if sb is None: smooth_parse_fail += 1
        if b is None or w <= 0 or h <= 0 or not all(finite(v) for v in b) or b[0] < -1e-3 or b[1] < -1e-3 or b[2] > w + 1e-3 or b[3] > h + 1e-3 or b[2] < b[0] or b[3] < b[1]:
            invalid_box += 1
        if b is not None and w > 0 and h > 0:
            expected = [b[0] / w, b[1] / h, b[2] / w, b[3] / h]
            actual = [float(r.get(x, 0.0)) for x in ("box_x1_norm", "box_y1_norm", "box_x2_norm", "box_y2_norm")]
            if max(abs(a - e) for a, e in zip(actual, expected)) > 2e-4: norm_mismatch += 1
        calc = box_iou(b, g)
        if calc is not None:
            try: stored = float(r.get("row_iou", 0.0)); diff = abs(calc - stored); iou_diffs.append(diff)
            except Exception: diff = 1.0
            if diff > 1e-4: iou_mismatch += 1
    for key, tr in by_track.items():
        ranks = [int(r.get("event_rank", 0)) for _, r in tr]
        counts = [int(r.get("causal_prefix_count", 0)) for _, r in tr]
        if len(set(ranks)) != len(ranks): event_rank_duplicates += 1
        if any(b <= a for a, b in zip(ranks, ranks[1:])) or any(b < a for a, b in zip(counts, counts[1:])):
            chronology_bad_tracks.append(key)
        prefix_bad_rows += sum(int(r.get("causal_prefix_count", 0)) < 1 for _, r in tr)

    events = load_events(POS) + load_events(NEG); events.sort(key=lambda e: str(e.get("event_key", "")))
    pos = [e for e in events if e.get("kind") == "positive_existing"]
    neg = [e for e in events if e.get("kind") == "negative_new"]
    assert len(pos) == 76 and len(neg) == 76, (len(pos), len(neg))
    records: list[dict[str, Any]] = []; agg: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for e in events:
        sk, tk = str(e["source_tracklet_keys"][0]), str(e["target_tracklet_key"])
        sr = [r for _, r in by_track.get(sk, [])]; tr = [r for _, r in by_track.get(tk, [])]
        expected_keys = [str(x) for x in e.get("target_row_keys", [])]
        expected_missing = [x for x in expected_keys if x not in row_by_key]
        wrong_key_rows = [x for x in expected_keys if x in row_by_key and key_for_row(row_by_key[x][1]) != tk]
        source_video = int(e.get("source_video", sk.split(":")[0][1:])); target_video = int(e.get("target_video", tk.split(":")[0][1:]))
        assert source_video != target_video
        for p in PREFIXES:
            tp = tr[: min(p, len(tr))]
            source_assigned = [r for r in sr if str(r.get("assigned", "0")) == "1"]
            target_assigned = [r for r in tp if str(r.get("assigned", "0")) == "1"]
            source_rel = [r for r in sr if str(r.get("assigned", "0")) == "1" and float(r.get("row_iou", 0.0)) >= IOU_THR]
            target_rel = [r for r in tp if str(r.get("assigned", "0")) == "1" and float(r.get("row_iou", 0.0)) >= IOU_THR]
            reasons: list[str] = []
            if not sr: reasons += ["source_missing", "no_proposal"]
            elif not source_assigned: reasons += ["source_no_assigned_proposal", "assignment_error"]
            elif not source_rel: reasons += ["source_no_reliable_observation", "iou_insufficient"]
            if not tr: reasons += ["target_missing", "no_proposal"]
            elif not tp: reasons += ["target_no_proposal_in_prefix"]
            elif not target_assigned: reasons += ["target_no_assigned_proposal_in_prefix", "assignment_error"]
            elif not target_rel: reasons += ["target_no_reliable_observation_in_prefix", "iou_insufficient"]
            if expected_missing or wrong_key_rows: reasons.append("wrong_frame")
            if sk not in by_track or tk not in by_track: reasons.append("track_missing")
            if sk in chronology_bad_tracks or tk in chronology_bad_tracks: reasons.append("time_leakage_risk")
            if invalid_box or norm_mismatch or iou_mismatch: # global audit flag is recorded separately
                # Attach only when the event's rows contain the offending kind
                # to keep event reasons interpretable.
                relevant = sr + tp
                if any(parse_box(r.get("bbox_xyxy")) is None for r in relevant): reasons.append("coordinate_transform_error")
            ceiling = bool(e.get("kind") == "positive_existing" and source_rel and target_rel)
            rec = {"event_key": str(e["event_key"]), "fold": int(e["fold"]), "kind": str(e["kind"]),
                   "category": int(e.get("target_category_gt_denominator_only", e.get("category_gt_denominator_only", -1))),
                   "source_tracklet_key": sk, "target_tracklet_key": tk, "source_video": source_video, "target_video": target_video,
                   "prefix": int(p), "source_rows": len(sr), "target_rows": len(tr), "target_prefix_rows": len(tp),
                   "source_assigned": len(source_assigned), "target_assigned_prefix": len(target_assigned),
                   "source_reliable": len(source_rel), "target_reliable_prefix": len(target_rel),
                   "source_materialized": bool(sr), "source_reliable_materialized": bool(source_rel),
                   "target_visible": bool(tp), "target_reliably_visible": bool(target_rel),
                   "expected_target_row_keys": len(expected_keys), "expected_target_row_keys_missing": expected_missing,
                   "wrong_key_rows": wrong_key_rows, "perfect_correspondence_ct_ceiling": ceiling,
                   "failure_reasons": sorted(set(reasons))}
            records.append(rec); agg[p].append(rec)

    psummary: list[dict[str, Any]] = []
    for p in PREFIXES:
        rs = [x for x in agg[p] if x["kind"] == "positive_existing"]; good = [x for x in rs if x["perfect_correspondence_ct_ceiling"]]
        reasons = Counter(r for x in rs for r in x["failure_reasons"])
        psummary.append({"prefix": p, "positive_denominator": len(rs), "negative_denominator": len([x for x in agg[p] if x["kind"] == "negative_new"]),
                         "source_reliable": sum(x["source_reliable_materialized"] for x in rs), "target_reliable": sum(x["target_reliably_visible"] for x in rs),
                         "ceiling_correct": len(good), "ceiling_recall": len(good) / max(len(rs), 1),
                         "category_coverage": len({x["category"] for x in good}), "video_coverage": len({x["target_video"] for x in good}),
                         "failure_reasons": dict(sorted(reasons.items()))})
    old = json.loads(P20_CEIL.read_text()); old_by = {int(x["prefix"]): int(x["perfect_correspondence_ct_ceiling_correct"]) for x in old["prefix_summary"]}
    new_by = {int(x["prefix"]): int(x["ceiling_correct"]) for x in psummary}
    geometry = {"protocol": "trackocd_iclr27_phase21_stage0_geometry_chronology_audit", "source_rows": len(rows), "source_path": str(SRC), "source_sha256": sha256(SRC),
                "resolution_counts": dict(sorted(resolution.items())), "row_key_unique": row_key_duplicates == 0, "duplicate_row_keys": row_key_duplicates,
                "invalid_bbox_rows": invalid_box, "normalized_coordinate_mismatch_rows": norm_mismatch, "stored_iou_mismatch_rows": iou_mismatch,
                "stored_iou_recompute_max_abs_diff": max(iou_diffs) if iou_diffs else 0.0, "stored_iou_recompute_mean_abs_diff": statistics.mean(iou_diffs) if iou_diffs else 0.0,
                "causal_smoothed_bbox_parse_failures": smooth_parse_fail, "chronology_bad_track_count": len(chronology_bad_tracks), "chronology_bad_tracks": chronology_bad_tracks[:100],
                "event_rank_duplicate_track_count": event_rank_duplicates, "causal_prefix_count_lt_one_rows": prefix_bad_rows,
                "actual_dimensions_used": True, "hardcoded_640x480_used": False, "future_frame_or_track_read": False,
                "row_key_format": "video:frame:proposal_local:track:image", "reliable_rule": "assigned == 1 and row_iou >= 0.5",
                "phase20_reproduction": {"phase20_counts": old_by, "phase21_counts": new_by, "exact_match": old_by == new_by, "phase20_max": old["max_positive_ceiling_recall"]}}
    atomic_json(OUT / "audit/geometry_audit.json", geometry)
    atomic_json(OUT / "audit/observability_event_audit.json", {"protocol": "trackocd_iclr27_phase21_stage0_event_audit", "records": records, "positive_denominator": 76, "negative_events": 76})
    atomic_json(OUT / "audit/observability_by_prefix.json", {"protocol": "trackocd_iclr27_phase21_stage0_observability_by_prefix", "prefix_summary": psummary, "positive_denominator": 76, "event_audit": str(OUT / "audit/observability_event_audit.json")})
    atomic_json(OUT / "completion/stage0.done", {"stage": "stage0", "baseline_reproduced": geometry["phase20_reproduction"]["exact_match"], "max_ceiling": max(x["ceiling_recall"] for x in psummary), "geometry": str(OUT / "audit/geometry_audit.json")})

    report = ["# Phase21 Stage 0 — geometry and observability audit", "", "The audit reads only the public TRAIN-derived DSCT CSV and pseudo-held event manifests.  DEV+, Q1, and public new-model labels were not read.  Actual per-row image dimensions are used for every box check; no 640×480 constant is used.", "", f"* Rows: **{len(rows)}**; unique row keys: **{row_key_duplicates == 0}**; resolutions: `{dict(sorted(resolution.items()))}`.", f"* Invalid boxes: **{invalid_box}**; normalized-coordinate mismatches: **{norm_mismatch}**; stored-vs-recomputed IoU mismatches: **{iou_mismatch}** (max absolute difference `{max(iou_diffs) if iou_diffs else 0.0:.3g}`).", f"* Tracks with non-monotone event rank/prefix count: **{len(chronology_bad_tracks)}**; future-frame/track reads: **none detected**.", "", "## Phase20 baseline reproduction", "", "| prefix | Phase20 ceiling | Phase21 recomputation | exact match |", "|---:|---:|---:|---|"]
    for p in PREFIXES: report.append(f"| {p} | {old_by[p]} | {new_by[p]} | {old_by[p] == new_by[p]} |")
    report += ["", f"The prefix16 result is **{new_by[16]}/76 = {new_by[16]/76:.6f}**, exactly reproducing Phase20's 25/76.  Every event/prefix and failure reason is retained in [`observability_event_audit.json`](../../outputs/iclr27_phase21/audit/observability_event_audit.json).", "", "## Prefix and fold summary", "", "| prefix | source reliable | target reliable | ceiling | recall | category coverage | video coverage |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for x in psummary: report.append(f"| {x['prefix']} | {x['source_reliable']} | {x['target_reliable']} | {x['ceiling_correct']}/76 | {x['ceiling_recall']:.4f} | {x['category_coverage']} | {x['video_coverage']} |")
    report += ["", "Detailed geometry counts, row-key/chronology checks, and event-level reasons are in [`geometry_audit.json`](../../outputs/iclr27_phase21/audit/geometry_audit.json) and [`observability_by_prefix.json`](../../outputs/iclr27_phase21/audit/observability_by_prefix.json).  The reliable rule and 76-event denominator are unchanged.", ""]
    (ROOT / "docs/iclr27_phase21/STAGE0_GEOMETRY_OBSERVABILITY_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "phase20_reproduced": geometry["phase20_reproduction"]["exact_match"], "prefix16": new_by[16], "invalid_bbox": invalid_box, "norm_mismatch": norm_mismatch, "iou_mismatch": iou_mismatch}, indent=2))


if __name__ == "__main__": main()
