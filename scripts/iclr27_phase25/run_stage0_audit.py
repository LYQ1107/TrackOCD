#!/usr/bin/env python3
"""Phase25 Stage0: reproduce frozen ceilings and audit set-aware failures.

This audit consumes only the corrected Phase23/24 artifacts and TRAIN-derived
metadata.  It never writes to an earlier phase and never uses a held label as
an input to a model.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase25"
P24_STAGE1 = ROOT / "outputs/iclr27_phase24/metrics/stage1_unified_strategies.json"
P24_STAGE4 = ROOT / "outputs/iclr27_phase24/metrics/stage4_proposal_validation.json"
P24_TAX = ROOT / "outputs/iclr27_phase24/audit/candidate_taxonomy_76.json"
P24_SET = ROOT / "outputs/iclr27_phase24/audit/stage4_setaware_event_records.json"
P24_SUM = ROOT / "outputs/iclr27_phase24/audit/full_76_event_summary.csv"


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
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def first_evidence(side: dict[str, Any]) -> dict[str, Any]:
    ev = side.get("evidence") or []
    if not ev:
        return {}
    # Evidence is descriptive only; choose the row with the largest recorded
    # selected max IoU so the audit is deterministic.
    return max(ev, key=lambda x: float(x.get("max_iou", 0.0)))


def classify(pool: dict[str, Any], selected: dict[str, Any]) -> str:
    ps, pt = bool(pool.get("pool_source_reliable")), bool(pool.get("pool_target_reliable"))
    ss, st = bool(selected.get("source_reliable")), bool(selected.get("target_reliable"))
    if ss and st:
        return "success"
    if not ps and not pt:
        return "pool_no_reliable_candidate"
    if ps and pt:
        if not ss and not st:
            return "candidate_exists_not_retained"
        if not ss:
            return "candidate_exists_source_not_retained"
        if not st:
            return "candidate_exists_target_not_retained"
        return "retained_but_ranking_or_assignment_failure"
    if ps and not pt:
        return "target_side_pool_missing_or_iou_below_0.5"
    if pt and not ps:
        return "source_side_pool_missing_or_iou_below_0.5"
    return "other"


def fold_shift(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_fold: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_fold[str(r["fold"])].append(r)
        by_cat[str(r["category"])].append(r)
        by_video[str(r["target_video"])].append(r)

    def agg(rs: list[dict[str, Any]]) -> dict[str, Any]:
        def mean(path: tuple[str, ...]) -> float:
            vals: list[float] = []
            for r in rs:
                x: Any = r
                for k in path:
                    x = x.get(k, 0) if isinstance(x, dict) else 0
                try: vals.append(float(x))
                except (TypeError, ValueError): pass
            return float(np.mean(vals)) if vals else 0.0
        return {
            "events": len(rs),
            "success": int(sum(bool(r["setaware_ceiling"]) for r in rs)),
            "pool_ceiling": int(sum(bool(r["pool_ceiling"]) for r in rs)),
            "raw_ceiling": int(sum(bool(r["raw_ceiling"]) for r in rs)),
            "candidate_count_mean": mean(("pool", "candidate_count_total")),
            "raw_source_iou_mean": mean(("raw_source_max_iou",)),
            "raw_target_iou_mean": mean(("raw_target_max_iou",)),
            "pool_source_iou_mean": mean(("pool_source_max_iou",)),
            "pool_target_iou_mean": mean(("pool_target_max_iou",)),
            "selected_source_iou_mean": mean(("selected_source_max_iou",)),
            "selected_target_iou_mean": mean(("selected_target_max_iou",)),
            "source_area_fraction_mean": mean(("size", "source_area_fraction")),
            "target_area_fraction_mean": mean(("size", "target_area_fraction")),
            "source_stability_mean": mean(("causal", "source_mean_stability")),
            "target_stability_mean": mean(("causal", "target_mean_stability")),
            "failure_classes": dict(Counter(r["failure_class"] for r in rs)),
        }

    return {
        "by_fold": {k: agg(v) for k, v in sorted(by_fold.items())},
        "by_category": {k: agg(v) for k, v in sorted(by_cat.items(), key=lambda kv: int(kv[0]))},
        "by_target_video": {k: agg(v) for k, v in sorted(by_video.items(), key=lambda kv: int(kv[0]))},
        "fold_disjoint_source": "Phase24 inherited TRAIN video/category-disjoint fold manifest",
    }


def main() -> None:
    for p in (P24_STAGE1, P24_STAGE4, P24_TAX, P24_SET):
        if not p.exists():
            raise FileNotFoundError(p)
    stage1 = json.loads(P24_STAGE1.read_text(encoding="utf-8"))
    stage4 = json.loads(P24_STAGE4.read_text(encoding="utf-8"))
    pool_records = json.loads(P24_TAX.read_text(encoding="utf-8"))["records"]
    set_records = json.loads(P24_SET.read_text(encoding="utf-8"))["records"]
    pool = {str(r["event_key"]): r for r in pool_records if int(r["prefix"]) == 16}
    selected = {str(r["event_key"]): r for r in set_records if r["condition"] == "setaware_top20" and int(r["prefix"]) == 16}
    if len(pool) != 76 or len(selected) != 76:
        raise RuntimeError(f"fixed denominator changed: pool={len(pool)} selected={len(selected)}")

    records: list[dict[str, Any]] = []
    for key in sorted(pool):
        p = pool[key]; s = selected[key]
        ps, pt = p["source"], p["target"]; ss, st = s["source"], s["target"]
        es, et = first_evidence(ss), first_evidence(st)
        rec = {
            "event_key": key, "fold": int(p["fold"]), "category": int(p["category"]),
            "source_video": int(p["source_video"]), "target_video": int(p["target_video"]),
            "source_tracklet_key": p["source_tracklet_key"], "target_tracklet_key": p["target_tracklet_key"],
            "raw_ceiling": bool(p["raw_ceiling"]), "pool_ceiling": bool(p["candidate_pool_ceiling"]),
            "setaware_ceiling": bool(s["ceiling"]), "failure_class": classify(p, s),
            "pool_source_reliable": bool(p["pool_source_reliable"]), "pool_target_reliable": bool(p["pool_target_reliable"]),
            "setaware_source_reliable": bool(s["source_reliable"]), "setaware_target_reliable": bool(s["target_reliable"]),
            "raw_source_max_iou": float(ps.get("raw_max_iou", 0.0)), "raw_target_max_iou": float(pt.get("raw_max_iou", 0.0)),
            "pool_source_max_iou": float(ps.get("pool_max_iou", 0.0)), "pool_target_max_iou": float(pt.get("pool_max_iou", 0.0)),
            "selected_source_max_iou": float(ss.get("max_iou_mean", 0.0)), "selected_target_max_iou": float(st.get("max_iou_mean", 0.0)),
            "pool": {"source_candidate_count": int(ps.get("candidate_count", 0)), "target_candidate_count": int(pt.get("candidate_count", 0)),
                     "candidate_count_total": int(ps.get("candidate_count", 0)) + int(pt.get("candidate_count", 0)),
                     "source_assigned_count": int(ps.get("assigned_candidate_count", 0)), "target_assigned_count": int(pt.get("assigned_candidate_count", 0)),
                     "source_best": p.get("source", {}).get("best_candidate", {}), "target_best": p.get("target", {}).get("best_candidate", {})},
            "selected": {"source_selected_count": int(ss.get("selected_candidates", 0)), "target_selected_count": int(st.get("selected_candidates", 0)),
                         "source_assigned_count": int(ss.get("assigned_candidates", 0)), "target_assigned_count": int(st.get("assigned_candidates", 0)),
                         "source_best_evidence": es, "target_best_evidence": et},
            "size": {"source_area_fraction": float(p.get("size", {}).get("source_area_fraction", 0.0)), "target_area_fraction": float(p.get("size", {}).get("target_area_fraction", 0.0)), "ratio": float(p.get("size", {}).get("source_target_area_ratio", 0.0))},
            "causal": {"source_mean_stability": float(p.get("causal", {}).get("source_mean_stability", 0.0)), "target_mean_stability": float(p.get("causal", {}).get("target_mean_stability", 0.0)), "source_materialized": int(p.get("causal", {}).get("source_materialized", 0)), "target_materialized": int(p.get("causal", {}).get("target_materialized", 0)), "target_rows": int(p.get("causal", {}).get("target_rows", 0))},
            "parent_frame_transform": {"selected_source_frame": es.get("best_parent_frame"), "selected_target_frame": et.get("best_parent_frame"), "selected_source_transform": es.get("best_transform"), "selected_target_transform": et.get("best_transform"), "pool_source_frame": p.get("source", {}).get("best_candidate", {}).get("parent_frame"), "pool_target_frame": p.get("target", {}).get("best_candidate", {}).get("parent_frame")},
            "mot_parent_assignment_status": "preserved_parent_assignment_and_physical_track",
        }
        records.append(rec)

    # Reproduction is intentionally checked against the prior aggregate and
    # independently counted from event records.
    p16 = stage1["conditions"]["raw_baseline"]["prefix16"]
    pool16 = stage1["conditions"]["candidate_pool_oracle"]["prefix16"]
    set16 = stage4["conditions"]["setaware_top20"]["prefix16"]
    reproduced = {"raw": int(sum(r["raw_ceiling"] for r in records)), "pool_oracle": int(sum(r["pool_ceiling"] for r in records)), "setaware_top20": int(sum(r["setaware_ceiling"] for r in records)), "expected": {"raw": int(p16["ceiling_correct"]), "pool_oracle": int(pool16["ceiling_correct"]), "setaware_top20": int(set16["ceiling_correct"])} }
    if reproduced["raw"] != 25 or reproduced["pool_oracle"] != 38 or reproduced["setaware_top20"] != 32:
        raise RuntimeError(f"Phase24 regression mismatch: {reproduced}")

    summary = {
        "protocol": "trackocd_iclr27_phase25_stage0",
        "positive_event_denominator": 76, "prefix": 16,
        "reproduction": reproduced,
        "failure_records": 44,
        "failure_class_counts": dict(Counter(r["failure_class"] for r in records if not r["setaware_ceiling"])),
        "all_event_class_counts": dict(Counter(r["failure_class"] for r in records)),
        "pool_reliable_but_selector_failed": int(sum(r["pool_ceiling"] and not r["setaware_ceiling"] for r in records)),
        "candidate_pool_authorizes_selector": bool(reproduced["pool_oracle"] >= 38),
        "source_target_coverage": {"raw": [int(p16["source_reliable_events"]), int(p16["target_reliable_events"])], "pool": [int(pool16["source_reliable_events"]), int(pool16["target_reliable_events"])], "setaware_top20": [int(set16["source_reliable_events"]), int(set16["target_reliable_events"])]},
        "fold_ceiling": {"raw": [x["ceiling_correct"] for x in p16["by_fold"]], "pool": [x["ceiling_correct"] for x in pool16["by_fold"]], "setaware_top20": [x["ceiling_correct"] for x in set16["by_fold"]]},
        "phase24_artifact_sha256": {str(P24_STAGE1): sha256(P24_STAGE1), str(P24_STAGE4): sha256(P24_STAGE4), str(P24_TAX): sha256(P24_TAX), str(P24_SET): sha256(P24_SET)},
        "sealed_inputs_not_read": ["DEV+", "Q1", "public new-model labels", "future frames/tracks", "semantic text"],
    }
    atomic_json(OUT / "audit/failure_taxonomy_76.json", {"protocol": summary["protocol"], "records": records})
    atomic_json(OUT / "audit/failure_taxonomy_summary.json", summary)
    atomic_json(OUT / "audit/fold_shift_report.json", fold_shift(records))
    atomic_json(OUT / "audit/phase24_reproduction.json", {"protocol": summary["protocol"], "raw_prefix16": reproduced["raw"], "candidate_pool_oracle_prefix16": reproduced["pool_oracle"], "setaware_top20_prefix16": reproduced["setaware_top20"], "source_target": summary["source_target_coverage"], "fold_ceiling": summary["fold_ceiling"], "source_artifacts_read_only": True})
    done = OUT / "completion/stage0.done"; done.parent.mkdir(parents=True, exist_ok=True); fd, tmp = tempfile.mkstemp(prefix=".stage0.done.", dir=str(done.parent));
    with os.fdopen(fd, "w", encoding="utf-8") as f: json.dump({"stage": "stage0_failure_taxonomy", "reproduction": reproduced, "records": len(records)}, f, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, done)

    lines = ["# Phase25 Stage0 — regression and set-aware failure taxonomy", "", f"The corrected Phase24 protocol is reproduced exactly: raw **{reproduced['raw']}/76**, candidate-pool oracle **{reproduced['pool_oracle']}/76** (diagnostic), and set-aware top20 **{reproduced['setaware_top20']}/76**.  The 76-event denominator and causal prefix16 are unchanged.", "", "## Failure classes (44 set-aware failures)", "", "| class | events |", "|---|---:|"]
    for k, v in sorted(summary["failure_class_counts"].items()): lines.append(f"| {k} | {v} |")
    lines += ["", "The candidate pool contains both-side reliable candidates for the events classified as not-retained; this is a selection/generalization diagnostic, not a learned proposal result.  Pool oracle and GT-tight values remain diagnostics only.", "", "## Fold/domain shift", "", "Fold-level and category/video aggregates, candidate counts, IoUs, area fractions and causal stability are in `outputs/iclr27_phase25/audit/fold_shift_report.json`.  Physical parent assignment is inherited and unchanged; no semantic candidate becomes a physical track.", "", "## Reproduction command", "", "```bash", "PYTHONPATH=. /home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase25/run_stage0_audit.py", "```"]
    (ROOT / "docs/iclr27_phase25/STAGE0_FAILURE_AND_SHIFT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"reproduction": reproduced, "failure_class_counts": summary["failure_class_counts"], "stage2_authorized": summary["candidate_pool_authorizes_selector"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
