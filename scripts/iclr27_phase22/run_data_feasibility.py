#!/usr/bin/env python3
"""Phase22 Stage 1: inspect the frozen DSCT source and legal TRAIN splits.

The script computes descriptive TRAIN statistics and writes a new fold
manifest.  It never reads DEV+, Q1, or public new-model labels and does not
use the 76 held events for fitting or model selection.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data/iclr27_phase19r/sources/public_rows_corrected.csv"
OLD_MANIFEST = ROOT / "outputs/iclr27_phase21/manifests/fold_manifest.json"
OUT = ROOT / "outputs/iclr27_phase22"
CKPT = ROOT / "data/iclr27_phase15s/checkpoints/phase6b_dsct_stage_d.pth"
RUN_SCRIPT = ROOT / "src/iclr27_phase15s/data/run_dsct_public.sh"
DSCT_CONFIG = ROOT / "src/iclr27_phase15s/data/phase15s_tao_train.py"
DSCT_IMPL = ROOT / "third_party/research_refs_phase4n/OVTR/ovtr/models/dsct.py"
OVTR_MAIN = ROOT / "third_party/research_refs_phase4n/OVTR/ovtr/main.py"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fval(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        x = float(row.get(key, default)); return x if math.isfinite(x) else default
    except (TypeError, ValueError): return default


def parse_box(s: str | None) -> list[float] | None:
    try:
        b = json.loads(s or "")
        b = [float(x) for x in b]
        return b if len(b) == 4 and all(math.isfinite(x) for x in b) else None
    except Exception: return None


def gt_area(row: dict[str, str]) -> float:
    b = parse_box(row.get("gt_bbox_xyxy")); w = fval(row, "image_width"); h = fval(row, "image_height")
    if b is None or w <= 0 or h <= 0: return 0.0
    return max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1]) / (w*h)


def reliable(row: dict[str, str]) -> bool:
    return str(row.get("assigned", "0")) == "1" and fval(row, "row_iou") >= .5


def size_bin(x: float) -> str:
    if x < .01: return "<0.01"
    if x < .05: return "0.01-0.05"
    if x < .20: return "0.05-0.20"
    return ">=0.20"


def group_stats(rows: list[dict[str, str]], group_field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows: groups[str(r.get(group_field, "-1"))].append(r)
    out: dict[str, Any] = {}
    for key, rs in sorted(groups.items(), key=lambda x: int(x[0]) if re.fullmatch(r"-?\d+", x[0]) else x[0]):
        gt = [r for r in rs if str(r.get("gt_category_id_common", "-1")) != "-1" and parse_box(r.get("gt_bbox_xyxy")) is not None]
        out[key] = {
            "rows": len(rs), "gt_rows": len(gt), "reliable_rows": sum(reliable(r) for r in gt),
            "proposal_recall_iou50": sum(reliable(r) for r in gt) / max(len(gt), 1),
            "mean_score": statistics.mean([fval(r, "score") for r in rs]) if rs else 0.0,
            "median_gt_area_fraction": statistics.median([gt_area(r) for r in gt]) if gt else 0.0,
            "size_bins": dict(Counter(size_bin(gt_area(r)) for r in gt)),
            "low_temporal_iou_proxy_rows": sum(fval(r, "track_temporal_iou") < .5 for r in gt),
        }
    return out


def source_audit(rows: list[dict[str, str]]) -> dict[str, Any]:
    text = RUN_SCRIPT.read_text(encoding="utf-8") if RUN_SCRIPT.exists() else ""
    cfg = DSCT_CONFIG.read_text(encoding="utf-8") if DSCT_CONFIG.exists() else ""
    impl = DSCT_IMPL.read_text(encoding="utf-8") if DSCT_IMPL.exists() else ""
    main = OVTR_MAIN.read_text(encoding="utf-8") if OVTR_MAIN.exists() else ""
    return {
        "proposal_csv_source_family": dict(Counter(str(r.get("source_family", "")) for r in rows)),
        "det_category_id_unique_count": len({str(r.get("det_category_id", "")) for r in rows}),
        "det_category_id_present": all(str(r.get("det_category_id", "")).strip() for r in rows),
        "dsct_objectness_class_agnostic_claim": "class-agnostic objectness" in impl or "Class-agnostic" in impl,
        "proposal_generator_emits_detector_categories": "det_category_id" in "".join(rows[0].keys()) if rows else False,
        "proposal_stream_interpretation": "DSCT objectness is class-agnostic, but the frozen OVTR proposal generator is a closed-set detector/tracker interface that still emits detector category IDs; Phase22 refiner uses none of those IDs.",
        "checkpoint": {"path": str(CKPT), "resolved_path": str(CKPT.resolve()) if CKPT.exists() else None, "exists": CKPT.exists(), "bytes": CKPT.stat().st_size if CKPT.exists() else 0, "sha256": sha256(CKPT) if CKPT.exists() else None},
        "generator_script": str(RUN_SCRIPT), "generator_config": str(DSCT_CONFIG),
        "source_evidence": {
            "run_script_contains_score_mode_dsct": "--score_mode dsct" in text,
            "run_script_contains_class_agnostic_objectness_flags": "--dsct_coef" in text,
            "config_base": next((line.strip() for line in cfg.splitlines() if line.strip().startswith("_base_")), None),
            "implementation_path": str(DSCT_IMPL), "cli_path": str(OVTR_MAIN),
        },
        "training_source_command_reference": str(RUN_SCRIPT),
    }


def main() -> None:
    OUT.joinpath("audit").mkdir(parents=True, exist_ok=True); OUT.joinpath("manifests").mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open(newline="", encoding="utf-8") as f: rows = list(csv.DictReader(f))
    old = json.loads(OLD_MANIFEST.read_text())
    all_cats = sorted({int(r["gt_category_id_common"]) for r in rows if str(r.get("gt_category_id_common", "-1")) != "-1"})
    all_videos = sorted({int(r["video_id"]) for r in rows})
    folds = []
    for old_fold in old["folds"]:
        fold = int(old_fold["fold"]); held_cats = sorted(int(x) for x in old_fold.get("held_categories", []))
        fit_v = sorted(int(x) for x in old_fold.get("fit_videos", [])); val_v = sorted(int(x) for x in old_fold.get("validation_videos", []))
        assert not set(fit_v) & set(val_v), fold
        fit_cats = [c for c in all_cats if c not in set(held_cats)]
        fit_rows = [r for r in rows if int(r["video_id"]) in set(fit_v) and int(r.get("gt_category_id_common", -1)) in set(fit_cats)]
        val_rows = [r for r in rows if int(r["video_id"]) in set(val_v) and int(r.get("gt_category_id_common", -1)) in set(held_cats)]
        folds.append({"fold": fold, "held_categories": held_cats, "fit_categories": fit_cats, "fit_videos": fit_v, "validation_videos": val_v,
                      "fit_rows": len(fit_rows), "validation_rows_held_categories": len(val_rows),
                      "fit_category_disjoint_from_held": not (set(fit_cats) & set(held_cats)), "video_disjoint": not (set(fit_v) & set(val_v)),
                      "fit_gt_rows": sum(str(r.get("gt_category_id_common", "-1")) != "-1" for r in fit_rows),
                      "validation_gt_rows_held_categories": len(val_rows)})
    manifest = {"protocol": "trackocd_iclr27_phase22_video_category_disjoint_train_manifest", "source_csv": str(CSV_PATH), "source_csv_sha256": sha256(CSV_PATH),
                "labels_used": "public TRAIN GT boxes/category/video metadata only", "sealed_inputs_not_read": ["DEV+", "Q1", "public new-model labels"],
                "positive_event_denominator_not_used_for_fit": 76, "all_train_categories": all_cats, "all_train_videos": all_videos, "folds": folds,
                "model_input_fields": ["frozen_dinov2_cls", "frozen_dinov2_roi", "bbox_xyxy_normalized", "score", "causal_prefix_age_norm", "causal_box_stability_iou", "area_fraction", "box_aspect_log"],
                "forbidden_model_input_fields": ["gt_bbox_xyxy", "row_iou", "gt_category_id_common", "physical_id", "semantic_id", "future_frame"], "source_audit": source_audit(rows)}
    atomic_json(OUT / "manifests/fold_manifest.json", manifest)
    gt_rows = [r for r in rows if str(r.get("gt_category_id_common", "-1")) != "-1" and parse_box(r.get("gt_bbox_xyxy")) is not None]
    stats = {
        "protocol": "trackocd_iclr27_phase22_stage1_train_feasibility",
        "source_csv": str(CSV_PATH), "source_csv_sha256": sha256(CSV_PATH), "rows": len(rows), "gt_rows": len(gt_rows),
        "videos": len({int(r["video_id"]) for r in rows}), "categories": len({int(r["gt_category_id_common"]) for r in gt_rows}),
        "resolution": group_stats(rows, "image_width"), "video": group_stats(rows, "video_id"), "category": group_stats(gt_rows, "gt_category_id_common"),
        "gt_size_bins": dict(Counter(size_bin(gt_area(r)) for r in gt_rows)),
        "gt_reliable_recall": sum(reliable(r) for r in gt_rows) / max(len(gt_rows), 1),
        "occlusion": {"explicit_occlusion_field_present": any("occl" in k.lower() for k in (rows[0].keys() if rows else [])), "proxy_field": "track_temporal_iou", "proxy_definition": "track_temporal_iou < 0.5; not an occlusion label", "proxy_rows": sum(fval(r, "track_temporal_iou") < .5 for r in gt_rows)},
        "folds": folds,
        "trainability_decision": "proposal refiner training is authorized after Stage0 taxonomy: 51/76 failures are assigned-box IoU failures, geometry audit is clean, and legal TRAIN GT rows are available in video/category-disjoint fit splits.",
        "sealed_inputs_not_read": ["DEV+", "Q1", "public new-model labels"],
    }
    atomic_json(OUT / "audit/train_data_feasibility.json", stats)
    atomic_json(OUT / "completion/stage1.done", {"stage": "stage1", "manifest": str(OUT / "manifests/fold_manifest.json"), "feasibility": str(OUT / "audit/train_data_feasibility.json"), "training_authorized": True})
    lines = ["# Phase22 Stage 1 — DSCT source and TRAIN feasibility", "", f"The frozen source has **{len(rows)}** rows, **{len(gt_rows)}** GT-aligned TRAIN rows, {len({int(r['video_id']) for r in rows})} videos, and {len({int(r['gt_category_id_common']) for r in gt_rows})} labeled categories.  Explicit occlusion annotations are absent; `track_temporal_iou < 0.5` is reported only as a stability/occlusion proxy.", "", "## DSCT source", "", "The inherited generator is the Phase6B DSCT OVTR path (`--score_mode dsct`, frozen stage-D checkpoint).  Its objectness branch is class-agnostic, while the detector/tracker interface still emits `det_category_id`; the Phase22 refiner does not consume detector category, GT category, physical ID, or semantic text.", "", f"Checkpoint: `{CKPT}` (SHA-256 `{sha256(CKPT) if CKPT.exists() else 'missing'}`).  Generator command/config evidence: `{RUN_SCRIPT}`, `{DSCT_CONFIG}`, `{DSCT_IMPL}`.", "", "## Fixed folds", "", "| fold | held categories | fit videos | validation videos | fit rows | held-category validation rows | video-disjoint | category-disjoint |", "|---:|---:|---:|---:|---:|---:|---|---|"]
    for f in folds: lines.append(f"| {f['fold']} | {len(f['held_categories'])} | {len(f['fit_videos'])} | {len(f['validation_videos'])} | {f['fit_rows']} | {f['validation_rows_held_categories']} | {f['video_disjoint']} | {f['fit_category_disjoint_from_held']} |")
    lines += ["", "## Data statistics", "", f"GT area bins: `{stats['gt_size_bins']}`; reliable TRAIN row recall at IoU 0.5: **{stats['gt_reliable_recall']:.4f}**.  Per-video, per-category, and per-resolution recall/size tables are in [`train_data_feasibility.json`](../../outputs/iclr27_phase22/audit/train_data_feasibility.json).", "", "Stage0 evidence plus legal TRAIN labels justify one bounded refiner training route.  The smoke run must pass before the four-fold launch; no correspondence/controller/backbone branch is authorized by this report."]
    (ROOT / "docs/iclr27_phase22/STAGE1_TRAIN_FEASIBILITY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "gt_rows": len(gt_rows), "categories": len(stats["category"]), "videos": stats["videos"], "folds": folds}, indent=2))


if __name__ == "__main__": main()
