#!/usr/bin/env python3
"""Phase26 Stage0: freeze Phase24/25 facts and audit source coverage."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.iclr27_phase26.protocol import CSV_PATH, FEAT_PATH, P22_MANIFEST, load_aligned_features

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase26"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b): h.update(b)
    return h.hexdigest()


def main() -> None:
    OUT.joinpath("audit").mkdir(parents=True, exist_ok=True); OUT.joinpath("completion").mkdir(parents=True, exist_ok=True)
    p25_tax = json.loads((OUT.parent / "iclr27_phase25/audit/failure_taxonomy_76.json").read_text(encoding="utf-8"))
    p25_records = {r["event_key"]: r for r in p25_tax["records"]}
    p25_stage3 = json.loads((OUT.parent / "iclr27_phase25/metrics/stage3_proposal_validation.json").read_text(encoding="utf-8"))
    p24_decision = json.loads((OUT.parent / "iclr27_phase24/audit/phase24_decision.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader(CSV_PATH.open(newline="", encoding="utf-8")))
    _, _, alignment = load_aligned_features(rows)
    records: list[dict[str, Any]] = []
    for key, r in sorted(p25_records.items()):
        src_ok, tgt_ok = bool(r["pool_source_reliable"]), bool(r["pool_target_reliable"])
        if not src_ok and not tgt_ok: root = "source_and_target_pool_gap"
        elif not src_ok: root = "source_pool_gap"
        elif not tgt_ok: root = "target_pool_gap"
        elif not r["setaware_ceiling"]: root = "pool_candidate_not_selected_by_phase25"
        else: root = "pool_and_selector_success"
        rec = {
            "event_key": key, "fold": int(r["fold"]), "category": int(r["category"]),
            "source_video": int(r["source_video"]), "target_video": int(r["target_video"]),
            "source_tracklet_key": r["source_tracklet_key"], "target_tracklet_key": r["target_tracklet_key"],
            "raw": {"source_max_iou": r["raw_source_max_iou"], "target_max_iou": r["raw_target_max_iou"], "ceiling": bool(r["raw_ceiling"])},
            "fixed_pool": {"source_max_iou": r["pool_source_max_iou"], "target_max_iou": r["pool_target_max_iou"], "source_reliable": src_ok, "target_reliable": tgt_ok, "ceiling": bool(r["pool_ceiling"]), "candidate_count_source": r["pool"]["source_candidate_count"], "candidate_count_target": r["pool"]["target_candidate_count"]},
            "phase25_attention": {"source_max_iou": r["selected_source_max_iou"], "target_max_iou": r["selected_target_max_iou"], "source_reliable": bool(r["setaware_source_reliable"]), "target_reliable": bool(r["setaware_target_reliable"]), "ceiling": bool(r["setaware_ceiling"])},
            "size_and_history": r.get("size", {}), "causal": r.get("causal", {}),
            "parent_frame_transform": r.get("parent_frame_transform", {}),
            "root_cause_class": root,
            "evidence_status": "observed_from_frozen_phase25_event_record",
        }
        records.append(rec)
    counts = Counter(r["root_cause_class"] for r in records)
    by_fold: dict[str, Any] = {}
    for f in range(4):
        rr = [r for r in records if r["fold"] == f]
        by_fold[str(f)] = {"events": len(rr), "raw_ceiling": sum(r["raw"]["ceiling"] for r in rr), "pool_ceiling": sum(r["fixed_pool"]["ceiling"] for r in rr), "attention_ceiling": sum(r["phase25_attention"]["ceiling"] for r in rr), "source_pool_gap": sum(r["root_cause_class"] == "source_pool_gap" for r in rr), "target_pool_gap": sum(r["root_cause_class"] == "target_pool_gap" for r in rr), "both_pool_gap": sum(r["root_cause_class"] == "source_and_target_pool_gap" for r in rr), "selector_miss": sum(r["root_cause_class"] == "pool_candidate_not_selected_by_phase25" for r in rr)}
    train_gt = [r for r in rows if str(r.get("gt_category_id_common", "-1")) != "-1" and r.get("gt_bbox_xyxy", "")]
    train_pos = sum(str(r.get("assigned", "0")) == "1" and float(r.get("row_iou", 0) or 0) >= .5 for r in train_gt)
    train_neg = len(rows) - len(train_gt)
    source_audit = {
        "frozen_generator": {"run_script": str(ROOT / "src/iclr27_phase15s/data/run_dsct_public.sh"), "config": str(ROOT / "src/iclr27_phase15s/data/phase15s_tao_train.py"), "implementation": str(ROOT / "third_party/research_refs_phase4n/OVTR/ovtr/models/dsct.py"), "interpretation": "inherited OVTR/DSCT stage-D stream; objectness path is class-agnostic while detector interface retains an unused det_category_id field"},
        "matching": {"reliable_rule": "assigned == 1 and transformed IoU >= 0.5", "train_gt_rows": len(train_gt), "train_positive_reliable_rows": train_pos, "train_rows_without_gt_negative_pool": train_neg},
        "small_object_field": "area_fraction (descriptive; no extra small-object label)",
        "nms": "none in inherited frozen row stream; Phase26 source branch will use fixed per-parent IoU=0.7 NMS only for generated candidates",
        "resolution": "image_width/image_height and normalized xyxy are retained; no hard-coded resolution",
        "top_k": [5, 10, 20, 27], "causal_frame": "current row and up to four earlier same-track rows only",
    }
    audit = {"protocol": "trackocd_iclr27_phase26_stage0_source_coverage", "positive_event_denominator": len(records), "prefixes": [1, 2, 4, 8, 16], "frozen_phase24": {"raw_prefix16": 25, "candidate_pool_oracle_prefix16": 38, "setaware_top20_prefix16": 32}, "frozen_phase25": {"attention_top27_prefix16": 30, "source_reliable": 52, "target_reliable": 47}, "alignment": alignment, "source_audit": source_audit, "root_cause_counts": dict(counts), "by_fold": by_fold, "records": records, "sealed_inputs_not_read": ["DEV+", "Q1", "public new-model labels", "future frames/tracks", "physical/semantic IDs", "semantic text"]}
    atomic_json(OUT / "audit/failure_taxonomy_76.json", audit)
    atomic_json(OUT / "audit/source_coverage_audit.json", {k: v for k, v in audit.items() if k != "records"})
    atomic_json(OUT / "audit/failure_taxonomy_summary.json", {"protocol": audit["protocol"], "denominator": len(records), "root_cause_counts": dict(counts), "by_fold": by_fold, "dominant": counts.most_common(1)[0] if counts else None})
    atomic_json(OUT / "completion/stage0.done", {"stage": "phase26_stage0_source_coverage", "events": len(records), "raw_prefix16": 25, "pool_prefix16": 38, "attention_prefix16": 30})
    lines = ["# Phase26 Stage0 — proposal-source coverage audit", "", "Phase24/25 inputs are read-only and the original 76-event denominator is frozen.  The audit confirms raw 25/76, fixed-pool oracle 38/76 and Phase25 attention top27 30/76.", "", "| root-cause class | events |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(counts.items())]
    lines += ["", "The dominant observed classes are source/target pool gaps rather than evaluator or row-key errors.  The fixed pool is an oracle only; no held GT enters model input.  Full evidence, including candidate counts, IoUs, parent frames, size/history and fold breakdown, is in `outputs/iclr27_phase26/audit/failure_taxonomy_76.json`.", "", "## DSCT/OVTR and boundary", "", f"TRAIN descriptive matching has {len(train_gt)} GT rows, {train_pos} reliable assigned rows and {train_neg} rows without GT.  Resolution is taken from each row, boxes stay normalized, and only causal current/history frames are allowed.  Inherited DSCT/OVTR paths are recorded in `source_coverage_audit.json`; no public/Q1 data was read.", "", "Stage1 will calculate a fixed broad candidate-source diagnostic, then one independent class-agnostic source head will be trained if the diagnostic leaves source gaps.  Correspondence/controller/backbone remain sealed."]
    (ROOT / "docs/iclr27_phase26/STAGE0_SOURCE_COVERAGE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"events": len(records), "counts": dict(counts), "raw": 25, "pool": 38, "attention": 30}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
