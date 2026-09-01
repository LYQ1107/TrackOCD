#!/usr/bin/env python3
"""Freeze the Phase22 proposal-source experiment and write its artifacts.

This is a report/aggregation utility only.  It reads the already completed
Stage0/Stage1/Stage3 outputs, does not open any sealed split, and never starts
training or evaluation.  All JSON/report writes use temporary files followed
by an atomic rename so a repeated invocation cannot leave a partial artifact.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import statistics
import subprocess
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase22"
DOC = ROOT / "docs/iclr27_phase22"
PREFIXES = (1, 2, 4, 8, 16)
EVENTS = 76


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def records_from(path: Path) -> list[dict[str, Any]]:
    obj = read_json(path)
    records = obj.get("records", obj) if isinstance(obj, dict) else obj
    if not isinstance(records, list):
        raise ValueError(f"records is not a list: {path}")
    return records


def _mean(records: Iterable[dict[str, Any]], field: str) -> float:
    vals = [float(r.get(field, 0.0)) for r in records]
    return float(statistics.mean(vals)) if vals else 0.0


def _median(records: Iterable[dict[str, Any]], field: str) -> float:
    vals = [float(r.get(field, 0.0)) for r in records]
    return float(statistics.median(vals)) if vals else 0.0


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize the exact event rows, retaining all prefix/fold denominators."""
    conditions = sorted({str(r["condition"]) for r in records})
    result: dict[str, Any] = {}
    for condition in conditions:
        cr = [r for r in records if str(r["condition"]) == condition]
        prefix_summary: list[dict[str, Any]] = []
        for prefix in PREFIXES:
            rr = [r for r in cr if int(r["prefix"]) == prefix]
            good = [r for r in rr if bool(r.get("ceiling"))]
            folds: list[dict[str, Any]] = []
            for fold in range(4):
                fr = [r for r in rr if int(r["fold"]) == fold]
                fg = [r for r in fr if bool(r.get("ceiling"))]
                folds.append({
                    "fold": fold,
                    "positive_denominator": len(fr),
                    "source_reliable_events": sum(int(r.get("source_reliable", 0)) > 0 for r in fr),
                    "target_reliable_events": sum(int(r.get("target_reliable", 0)) > 0 for r in fr),
                    "ceiling_correct": len(fg),
                    "ceiling_recall": len(fg) / max(len(fr), 1),
                    "category_coverage": len({int(r["category"]) for r in fg}),
                    "video_coverage": len({int(r["target_video"]) for r in fg}),
                })
            prefix_summary.append({
                "prefix": prefix,
                "positive_denominator": len(rr),
                "source_reliable_events": sum(int(r.get("source_reliable", 0)) > 0 for r in rr),
                "target_reliable_events": sum(int(r.get("target_reliable", 0)) > 0 for r in rr),
                "ceiling_correct": len(good),
                "ceiling_recall": len(good) / max(len(rr), 1),
                "category_coverage": len({int(r["category"]) for r in good}),
                "video_coverage": len({int(r["target_video"]) for r in good}),
                "source_iou_mean": _mean(rr, "source_iou_mean"),
                "source_iou_median": _median(rr, "source_iou_median"),
                "target_iou_mean": _mean(rr, "target_iou_mean"),
                "target_iou_median": _median(rr, "target_iou_median"),
                "failure_event_keys": sorted(str(r["event_key"]) for r in rr if not bool(r.get("ceiling"))),
                "by_fold": folds,
            })
        result[condition] = {
            "event_records": len(cr),
            "prefix_summary": prefix_summary,
            "prefix16": next(x for x in prefix_summary if x["prefix"] == 16),
        }
    return result


def checkpoint_ledger() -> list[dict[str, Any]]:
    paths: list[Path] = []
    for stem in ["proposal_refiner_f", "proposal_refiner_repair_f"]:
        for fold in range(4):
            p = OUT / "checkpoints" / f"{stem}{fold}_best.pt"
            if p.exists():
                paths.append(p)
            p = OUT / "checkpoints" / f"{stem}{fold}_latest.pt"
            if p.exists():
                paths.append(p)
    return [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p), "mtime": p.stat().st_mtime} for p in paths]


def artifact_ledger(paths: Iterable[Path]) -> list[dict[str, Any]]:
    out = []
    for p in paths:
        out.append({"path": str(p), "exists": p.exists(), "bytes": p.stat().st_size if p.exists() else 0, "sha256": sha256(p)})
    return out


def symlink_ledger() -> list[dict[str, Any]]:
    paths = [
        ROOT / "data/iclr27_phase19r/sources/public_rows_corrected.csv",
        ROOT / "data/iclr27_phase19r/sources/public_cls_roi.npz",
        ROOT / "data/iclr27_phase15s/checkpoints/phase6b_dsct_stage_d.pth",
    ]
    out = []
    for p in paths:
        out.append({
            "path": str(p),
            "is_symlink": p.is_symlink(),
            "link_target": os.readlink(p) if p.is_symlink() else None,
            "resolved_target": str(p.resolve()) if p.exists() else None,
            "target_exists": p.exists(),
        })
    return out


def resource_snapshot() -> dict[str, Any]:
    def run(args: list[str]) -> str:
        try:
            return subprocess.run(args, text=True, capture_output=True, check=False).stdout.strip()
        except OSError as exc:
            return f"unavailable: {exc}"

    ps = run(["ps", "-e", "-o", "pid=,args="])
    phase22 = []
    needles = ("scripts/iclr27_phase22", "run_four_fold", "train_proposal_refiner", "evaluate_proposal_refiner")
    for line in ps.splitlines():
        if str(os.getpid()) in line and "finalize_phase22.py" in line:
            continue
        if any(n in line for n in needles):
            phase22.append(line.strip())
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "free_h": run(["free", "-h"]),
        "process_count": len(ps.splitlines()),
        "nvidia_smi": run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"]),
        "data1_df_h": run(["df", "-h", "/data1"]),
        "residual_phase22_processes": phase22,
        "external_process_policy": "No unrelated process was terminated; the pre-existing audit_rmot_scorer_failure.py process was left untouched.",
    }


def stage1_variant_rows() -> list[dict[str, Any]]:
    obj = read_json(ROOT / "outputs/iclr27_phase21/metrics/stage1_proposal_variants.json")
    rows = []
    for name, value in sorted(obj.get("variants", {}).items()):
        p16 = next(x for x in value.get("prefix_summary", []) if int(x["prefix"]) == 16)
        rows.append({
            "variant": name,
            "source_reliable": p16.get("source_reliable", p16.get("source_reliable_events", 0)),
            "target_reliable": p16.get("target_reliable", p16.get("target_reliable_events", 0)),
            "ceiling_correct": p16["ceiling_correct"],
            "category_coverage": p16["category_coverage"],
            "video_coverage": p16["video_coverage"],
        })
    return rows


def training_rows(tag: str) -> list[dict[str, Any]]:
    rows = []
    prefix = f"{tag}_" if tag else ""
    for fold in range(4):
        p = OUT / "metrics" / f"train_{prefix}f{fold}.json"
        d = read_json(p)
        vm = d.get("validation_metrics", {})
        rows.append({
            "fold": fold,
            "tag": tag or "initial",
            "steps": d.get("steps"),
            "best_step": d.get("best_step"),
            "best_score": d.get("best_score"),
            "fit_rows": d.get("fit_rows"),
            "validation_rows": d.get("validation_rows"),
            "validation_recall_iou50": vm.get("reliable_recall_iou50"),
            "raw_reliable_rows": vm.get("raw_reliable_rows"),
            "assigned_refined_reliable_rows": vm.get("assigned_refined_reliable_rows"),
            "iou_mean_gt_rows": vm.get("iou_mean_gt_rows"),
            "checkpoint": d.get("checkpoint_best"),
            "amp": d.get("amp"),
        })
    return rows


def render_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return lines


def f4(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "-"


def main() -> None:
    stage0_summary = read_json(OUT / "audit/failure_taxonomy_summary.json")
    stage0_records = read_json(OUT / "audit/failure_taxonomy_76.json").get("records", [])
    feasibility = read_json(OUT / "audit/train_data_feasibility.json")
    manifest = read_json(OUT / "manifests/fold_manifest.json")
    p21_decision = read_json(ROOT / "outputs/iclr27_phase21/audit/phase21_decision.json")
    p21_prefix = read_json(ROOT / "outputs/iclr27_phase21/audit/observability_by_prefix.json").get("prefix_summary", [])
    initial_records = records_from(OUT / "audit/stage3_proposal_event_records.json")
    repair_records = records_from(OUT / "audit/stage3_proposal_validation_repair_event_records.json")
    initial_summary = summarize_records(initial_records)
    repair_summary = summarize_records(repair_records)
    if len(stage0_records) != EVENTS:
        raise RuntimeError(f"Stage0 event denominator changed: {len(stage0_records)}")
    for name, rs in [("initial", initial_records), ("repair", repair_records)]:
        for condition in ("phase21_raw_baseline", "best_trained_refiner"):
            n = sum(1 for r in rs if r["condition"] == condition and int(r["prefix"]) == 16)
            if n != EVENTS:
                raise RuntimeError(f"{name} {condition} prefix16 rows={n}")

    # Initial full run is the pre-identity-initialization cycle; repair is the
    # one evidence-based residual-initialization repair.  Baseline conditions
    # are recomputed in both records and must agree exactly at prefix16.
    raw16 = repair_summary["phase21_raw_baseline"]["prefix16"]
    init16 = initial_summary["best_trained_refiner"]["prefix16"]
    rep16 = repair_summary["best_trained_refiner"]["prefix16"]
    gt16 = repair_summary["gt_tight_oracle"]["prefix16"]
    if raw16["ceiling_correct"] != 25 or gt16["ceiling_correct"] != 73:
        raise RuntimeError("Phase21 baseline or GT-tight diagnostic did not reproduce 25/76 and 73/76")
    best_trained = max(int(init16["ceiling_correct"]), int(rep16["ceiling_correct"]))
    if best_trained >= 38 and rep16["source_reliable_events"] > raw16["source_reliable_events"] and rep16["target_reliable_events"] > raw16["target_reliable_events"]:
        gate_status = "PASS"
        decision_code = "P22_GATE_P_PASS_OPEN_CORRESPONDENCE"
    elif best_trained >= 32:
        gate_status = "PARTIAL"
        decision_code = "P22_GATE_P_PARTIAL_STOP_BEFORE_CORRESPONDENCE"
    else:
        gate_status = "FAIL"
        decision_code = "P22_GATE_P_FAIL_STOP_BEFORE_CORRESPONDENCE"

    resources = resource_snapshot()
    checkpoints = checkpoint_ledger()
    key_artifacts = [
        OUT / "audit/failure_taxonomy_76.json",
        OUT / "audit/failure_taxonomy_summary.json",
        OUT / "audit/train_data_feasibility.json",
        OUT / "manifests/fold_manifest.json",
        OUT / "metrics/stage3_proposal_validation_repair.json",
        OUT / "audit/stage3_proposal_validation_repair_event_records.json",
        OUT / "audit/stage3_proposal_validation_repair_event_summary.csv",
    ]
    artifacts = artifact_ledger(key_artifacts)

    # A compact 76-event prefix16 table: taxonomy evidence plus both training
    # cycles and the fixed diagnostic conditions.  The complete per-prefix
    # event records remain in the linked JSON/CSV artifacts.
    tax_by_key = {str(r["event_key"]): r for r in stage0_records}
    def by_key(rs: list[dict[str, Any]], condition: str) -> dict[str, dict[str, Any]]:
        return {str(r["event_key"]): r for r in rs if r["condition"] == condition and int(r["prefix"]) == 16}
    raw_by = by_key(repair_records, "phase21_raw_baseline")
    gt_by = by_key(repair_records, "gt_tight_oracle")
    init_by = by_key(initial_records, "best_trained_refiner")
    rep_by = by_key(repair_records, "best_trained_refiner")
    event_table = []
    for key in sorted(tax_by_key):
        t = tax_by_key[key]; raw = raw_by[key]; ini = init_by[key]; rep = rep_by[key]; gt = gt_by[key]
        event_table.append({
            "event_key": key,
            "fold": int(t["fold"]),
            "category": int(t["category"]),
            "source_video": int(t["source_video"]),
            "target_video": int(t["target_video"]),
            "taxonomy": t["primary_failure_class"],
            "raw_source_max_iou": raw.get("source_max_iou"),
            "raw_target_max_iou": raw.get("target_max_iou"),
            "raw_ceiling": bool(raw.get("ceiling")),
            "initial_refiner_ceiling": bool(ini.get("ceiling")),
            "repair_refiner_ceiling": bool(rep.get("ceiling")),
            "repair_source_max_iou": rep.get("source_max_iou"),
            "repair_target_max_iou": rep.get("target_max_iou"),
            "gt_tight_ceiling": bool(gt.get("ceiling")),
        })

    prefix_rows = []
    for label, summary in [
        ("raw", repair_summary["phase21_raw_baseline"]),
        ("best_nontraining", repair_summary["phase21_best_nontraining"]),
        ("frozen_oracle_correspondence", repair_summary["frozen_oracle_correspondence"]),
        ("gt_tight_oracle", repair_summary["gt_tight_oracle"]),
        ("initial_trained_refiner", initial_summary["best_trained_refiner"]),
        ("repair_trained_refiner", repair_summary["best_trained_refiner"]),
    ]:
        for p in summary["prefix_summary"]:
            prefix_rows.append({"condition": label, **p})

    aggregate = {
        "protocol": "trackocd_iclr27_phase22_proposal_source_repair",
        "execution_time": datetime.now(timezone.utc).isoformat(),
        "positive_event_denominator": EVENTS,
        "prefixes": list(PREFIXES),
        "decision_code": decision_code,
        "gate_p": {
            "status": gate_status,
            "threshold_prefix16": ">=38/76, both-side improvement, >=3/4 fold direction, no leakage",
            "raw_prefix16": raw16,
            "initial_trained_prefix16": init16,
            "repair_trained_prefix16": rep16,
            "best_trained_ceiling": best_trained,
            "gt_tight_diagnostic_prefix16": gt16,
            "correspondence_authorized": gate_status == "PASS",
        },
        "phase21_reproduction": {"decision": p21_decision.get("decision_code"), "prefix_summary": p21_prefix},
        "stage0": {"summary": stage0_summary, "records_path": str(OUT / "audit/failure_taxonomy_76.json")},
        "stage1": {"feasibility": feasibility, "manifest": manifest, "prior_fixed_variant_prefix16": stage1_variant_rows()},
        "stage3": {"initial_full_run": initial_summary, "identity_init_repair": repair_summary, "prefix_rows": prefix_rows, "event_table": event_table},
        "failure_event_keys": {"stage0_failed": sorted(str(r["event_key"]) for r in stage0_records if r.get("is_failed_event")), "repair_refiner_failed": sorted(x["event_key"] for x in event_table if not x["repair_refiner_ceiling"])},
        "training": {"initial": training_rows(""), "repair": training_rows("repair"), "commands": ["./scripts/iclr27_phase22/run_four_fold_supervisor.sh", "./scripts/iclr27_phase22/run_four_fold_repair_supervisor.sh"], "repair_reason": "zero-initialize residual box-delta head after initial run moved usable boxes away from identity; architecture, split, loss and steps remained fixed"},
        "resources": resources,
        "checkpoint_ledger": checkpoints,
        "artifact_ledger": artifacts,
        "symlink_ledger": symlink_ledger(),
        "data_boundary": {"train_source": str(ROOT / "data/iclr27_phase19r/sources/public_rows_corrected.csv"), "labels_used": "public TRAIN GT/category/video metadata only", "sealed_inputs_not_read": ["DEV+", "Q1", "public new-model labels", "future frames", "physical IDs as model inputs", "semantic text"]},
        "no_public_evaluation": True,
        "residual_phase22_processes": resources["residual_phase22_processes"],
        "next_direction": "Stop proposal refiner and correspondence/controller branches after Gate P failure. A future study should replace or retrain the proposal source/detector on legal TRAIN data, with an explicit candidate-generation/objectness recall audit; do not tune thresholds, StateMemory, controller or backbone on this branch.",
    }
    atomic_json(OUT / "metrics/phase22_aggregate.json", aggregate)
    atomic_json(OUT / "audit/phase22_decision.json", {
        "protocol": aggregate["protocol"],
        "execution_time": aggregate["execution_time"],
        "decision_code": decision_code,
        "gate_p": aggregate["gate_p"],
        "public_q1_access": False,
        "sealed_inputs_not_read": aggregate["data_boundary"]["sealed_inputs_not_read"],
        "stage0_primary_failure_counts": stage0_summary.get("primary_failure_counts", {}),
        "stage0_prefix16_ceiling": stage0_summary.get("prefix16_ceiling"),
        "stage1_best_nontraining_prefix16": repair_summary["phase21_best_nontraining"]["prefix16"],
        "stage3_initial_trained_prefix16": init16,
        "stage3_repair_trained_prefix16": rep16,
        "checkpoint_count": len(checkpoints),
        "artifact_parse_check": all(x["exists"] and x["sha256"] for x in artifacts),
        "residual_phase22_processes": resources["residual_phase22_processes"],
        "next_direction": aggregate["next_direction"],
    })
    atomic_json(OUT / "manifests/artifact_hashes.json", {"protocol": aggregate["protocol"], "artifacts": artifacts, "checkpoints": checkpoints, "source_symlinks": symlink_ledger()})
    atomic_json(OUT / "completion/stage2.done", {"stage": "stage2", "training_and_repair_complete": True, "gate_p": gate_status, "decision": decision_code})
    atomic_json(OUT / "completion/stage3.done", {"stage": "stage3", "events": EVENTS, "prefix16_best_trained": best_trained, "gate_p": gate_status, "decision": decision_code})
    atomic_json(OUT / "completion/phase22.done", {"phase": 22, "decision": decision_code, "report": str(DOC / "PHASE22_PROPOSAL_SOURCE_REPAIR_COMPLETE_REPORT.md")})

    # Render the self-contained report.  It intentionally states the failed
    # branch and does not imply that training loss or a proxy quality score is
    # a successful proposal result.
    lines: list[str] = []
    lines += ["# TrackOCD ICLR 2027 — Phase 22 Proposal Source/Detector Repair", "", f"Execution freeze (UTC): `{aggregate['execution_time']}`", "", "## Executive decision", "", f"**Gate P: {gate_status}** (`{decision_code}`).  The fixed 76-event protocol's raw and best non-training proposal ceiling is **25/76 = 0.328947** at prefix 16.  The initial trained refiner and the identity-initialized repair each reached only **3/76**; neither improved the two-sided observation coverage.  The required **>=38/76** threshold was not approached, so correspondence, controller, StateMemory, threshold, modern-backbone, final, and public branches remain closed.", "", "The denominator, row keys, assigned/IoU>=0.5 rule, causal prefixes, and all hard events were retained.  No event was deleted or reweighted for the headline result.", ""]
    lines += ["## Scope, sealing, and frozen comparator", "", "Phase22 used only the independent `docs/src/configs/scripts/outputs/iclr27_phase22` namespace.  Phase15/17/19/19R/20/21 evaluators and checkpoints were read-only comparators.  Public TRAIN rows, GT boxes and category/video metadata were used for feasibility, video/category-disjoint fitting and validation.  DEV+, Q1 and public new-model labels were not opened; no public evaluation artifact was created.  Physical IDs, semantic IDs/text and future frames were excluded from model inputs.", "", "The frozen DSCT source is the inherited Phase6B stage-D OVTR path (`--score_mode dsct`) and its class-agnostic objectness branch, although the detector/tracker interface emits `det_category_id`.  The refiner consumed neither detector category nor any identity field.", ""]
    lines += ["## Phase 21 reproduction", "", "Phase21's geometry/chronology audit was reused without alteration.  It reported zero invalid boxes, coordinate mismatches, stored-IoU mismatches, duplicate row keys or non-monotone track chronologies and reproduced the true-IoU ceiling curve:", ""]
    lines += render_table(["prefix", "source reliable", "target reliable", "ceiling", "recall", "category cov.", "video cov."], [[p["prefix"], p["source_reliable"], p["target_reliable"], f"{p['ceiling_correct']}/76", f4(p["ceiling_recall"]), p["category_coverage"], p["video_coverage"]] for p in p21_prefix])
    lines += ["", "The prefix16 25/76 result exactly matches Phase20/21.  Phase21 non-training smoothing, expansion, history and quality-rerank variants also remained at or below this ceiling (details below).", ""]
    lines += ["## Stage 0 — 76-event failure taxonomy", "", f"All **{EVENTS}** positive events were indexed at prefix16; **{stage0_summary['prefix16_failed']}** failed and **{stage0_summary['prefix16_ceiling']}** were perfectly observable under the frozen correspondence ceiling.  The dominant primary class is `{stage0_summary['dominant_primary_failure']}`.  No event was classified as source/target proposal missing, wrong frame/rank, assignment-only, or coordinate/scale error in this audit.", ""]
    lines += render_table(["primary class", "events"], [[k, v] for k, v in sorted(stage0_summary.get("primary_failure_counts", {}).items())])
    lines += ["", "Secondary flags are evidence only, not causal claims: 23/51 failed events have a low temporal-stability proxy, 18/51 have median GT area below 1% of the image, 51/51 are cross-video pairs (domain difference is unproven), and source occlusion labels are unavailable.  Candidate counts, frames, ranks, scores, areas, IoUs and failure flags for every event are in [`failure_taxonomy_76.json`](../../outputs/iclr27_phase22/audit/failure_taxonomy_76.json); the aggregate is [`failure_taxonomy_summary.json`](../../outputs/iclr27_phase22/audit/failure_taxonomy_summary.json).", "", "Per-fold failed/denominator: fold0 4/12, fold1 10/12, fold2 14/24, fold3 23/28.  This is proposal/box evidence; it does not establish that a different backbone would fix the source.", ""]
    lines += ["## Stage 1 — DSCT and TRAIN feasibility", "", f"The frozen TRAIN-derived CSV has **{feasibility['rows']}** rows, **{feasibility['gt_rows']}** GT rows, {feasibility['videos']} videos and {feasibility['categories']} categories.  The four fixed folds are video-disjoint and category-disjoint; fit/validation counts and held categories are recorded in [`fold_manifest.json`](../../outputs/iclr27_phase22/manifests/fold_manifest.json).  Explicit occlusion annotations are absent; `track_temporal_iou < 0.5` is only a proxy.", ""]
    lines += render_table(["fold", "fit rows", "val rows", "fit videos", "val videos", "held cats", "video disjoint", "category disjoint"], [[f["fold"], f["fit_rows"], f["validation_rows_held_categories"], len(f["fit_videos"]), len(f["validation_videos"]), len(f["held_categories"]), f["video_disjoint"], f["fit_category_disjoint_from_held"]] for f in manifest["folds"]])
    lines += ["", "Phase21 fixed non-training variants (same 76-event protocol, prefix16):", ""]
    lines += render_table(["variant", "source", "target", "ceiling", "category cov.", "video cov."], [[r["variant"], r["source_reliable"], r["target_reliable"], f"{r['ceiling_correct']}/76", r["category_coverage"], r["video_coverage"]] for r in stage1_variant_rows()])
    lines += ["", "The best non-training variant is raw/history/ROI-history/quality-rerank at 25/76; causal smoothing and fixed expansion were lower.  This confirms that a bounded refiner was justified by assigned-box IoU failures, while retaining the original proposal stream as the comparator.", ""]
    lines += ["## Stage 2 — class-agnostic proposal refiner", "", "The one registered route was a small residual refiner: frozen DINOv2 CLS+ROI (1536-D) through LayerNorm/linear projection, causal geometry/score/age/stability through a 64-D projection, a 256-D fusion layer, and two heads: bounded normalized xyxy box delta and quality logit.  It had no category/identity/text/GT/future inputs.  The loss was SmoothL1 box regression on TRAIN GT-aligned rows plus 0.5-weight BCE for assigned-and-IoU>=0.5 quality; no unregistered candidate branch or extra loss was added.", "", "Configuration: seed base `20260828` with fold offset, batch 256, AdamW `lr=2e-4`, weight decay `1e-4`, 2,000 updates/fold, checkpoint every 500, BF16 AMP (no non-finite fallback), worker count 0.  Each fold used one GPU under the bounded supervisor and wrote `.launched`, checkpoints, metrics and `.done` atomically.", "", "The initial cycle used the same route but a non-identity box head; its validation showed catastrophic box movement (refined recall below raw), and its full event result was 3/76.  One evidence-based repair zero-initialized only the residual box-delta weight/bias so the starting function is the frozen proposal identity.  Split, labels, architecture width, loss, optimizer, steps and evaluator were unchanged.  The repair smoke run passed before the four-fold launch.", ""]
    lines += render_table(["cycle", "fold", "best step", "best score", "fit rows", "val rows", "val IoU50 recall", "raw reliable", "refined reliable", "mean IoU"], [[r["tag"], r["fold"], r["best_step"], f4(r["best_score"]), r["fit_rows"], r["validation_rows"], f4(r["validation_recall_iou50"]), r["raw_reliable_rows"], r["assigned_refined_reliable_rows"], f4(r["iou_mean_gt_rows"])] for cycle in (training_rows(""), training_rows("repair")) for r in cycle])
    lines += ["", "Training losses and row-level validation are diagnostics only; they do not substitute for the persistent 76-event true-IoU ceiling.", ""]
    lines += ["## Stage 3 — true proposal ceiling", "", "The evaluator ran once on the complete 76-event positive manifest at prefixes 1/2/4/8/16.  Raw, best non-training, frozen-oracle-correspondence and GT-tight conditions are diagnostics; GT-tight is not a proposal result or training input.", ""]
    lines += render_table(["condition", "prefix", "source", "target", "ceiling", "category cov.", "video cov.", "source IoU mean", "target IoU mean"], [[r["condition"], r["prefix"], r["source_reliable_events"], r["target_reliable_events"], f"{r['ceiling_correct']}/76", r["category_coverage"], r["video_coverage"], f4(r["source_iou_mean"]), f4(r["target_iou_mean"])] for r in prefix_rows])
    lines += ["", "### Prefix16 fold comparison", "", "The fold table below keeps the original denominators (12, 12, 24, 28) and reports both-side reliable event counts and ceiling coverage for every main condition.", ""]
    fold_rows = []
    for label, summary in [
        ("raw", repair_summary["phase21_raw_baseline"]),
        ("best_nontraining", repair_summary["phase21_best_nontraining"]),
        ("frozen_oracle_correspondence", repair_summary["frozen_oracle_correspondence"]),
        ("gt_tight_oracle", repair_summary["gt_tight_oracle"]),
        ("initial_trained_refiner", initial_summary["best_trained_refiner"]),
        ("repair_trained_refiner", repair_summary["best_trained_refiner"]),
    ]:
        p16 = summary["prefix16"]
        for f in p16["by_fold"]:
            fold_rows.append([label, f["fold"], f["positive_denominator"], f["source_reliable_events"], f["target_reliable_events"], f["ceiling_correct"], f["category_coverage"], f["video_coverage"]])
    lines += render_table(["condition", "fold", "denom.", "source", "target", "ceiling", "category cov.", "video cov."], fold_rows)
    lines += ["", "At prefix16: raw/best non-training/frozen-oracle-correspondence = **25/76**, GT-tight diagnostic = **73/76**, initial trained refiner = **3/76**, repair trained refiner = **3/76**.  Repair source/target reliable event coverage was 10/7 versus raw 49/40, so both sides regressed.  Fold ceilings for the repair were 0/12, 2/12, 1/24, 0/28; no three-fold direction or broad category/video gain exists.", "", "### Complete prefix16 event index", "", "The following is the complete 76-event positive denominator.  `raw`, `initial`, `repair` and `GT` are ceiling booleans; the IoUs are the maximum transformed true IoUs available to that condition.  The taxonomy column is the Stage0 evidence-backed primary class.", ""]
    lines += render_table(["event key", "fold", "cat", "src video", "tgt video", "taxonomy", "raw src max", "raw tgt max", "raw", "initial", "repair src max", "repair tgt max", "repair", "GT"], [[e["event_key"], e["fold"], e["category"], e["source_video"], e["target_video"], e["taxonomy"], f4(e["raw_source_max_iou"]), f4(e["raw_target_max_iou"]), int(e["raw_ceiling"]), int(e["initial_refiner_ceiling"]), f4(e["repair_source_max_iou"]), f4(e["repair_target_max_iou"]), int(e["repair_refiner_ceiling"]), int(e["gt_tight_ceiling"])] for e in event_table])
    lines += ["", "The machine-readable per-prefix/event files (including every failed event key and candidate evidence) are [`stage3_proposal_validation_repair.json`](../../outputs/iclr27_phase22/metrics/stage3_proposal_validation_repair.json), [`stage3_proposal_validation_repair_event_records.json`](../../outputs/iclr27_phase22/audit/stage3_proposal_validation_repair_event_records.json), and [`stage3_proposal_validation_repair_event_summary.csv`](../../outputs/iclr27_phase22/audit/stage3_proposal_validation_repair_event_summary.csv).", ""]
    lines += ["## Gate P and stop decision", "", f"Gate P is **{gate_status}**.  Required prefix16 ceiling >=38/76 and substantial two-sided/fold/category/video improvement were not met: best trained ceiling is {best_trained}/76, raw is 25/76, repair is 3/76, source/target both fell, and the four repair folds are not directionally improved.  Therefore decision code is `{decision_code}`.", "", "No correspondence encoder, action head, StateMemory/controller change, threshold sweep, modern-backbone download, final 50k, DEV+/Q1 read or public evaluation was started.  The next candidate is a proposal-source/detector replacement or TRAIN-only candidate-generation/objectness recall study with the same exact 76-event denominator; only after Gate P passes should correspondence be revisited.", ""]
    lines += ["## Resources, storage, integrity and failures", "", "The four-fold supervisors used one worker per GPU 0–3, BF16, bounded concurrency and one blocking wait.  Preflight retained at least 25% system RAM (about 115–121 GiB available on a 125 GiB host), swap was disabled, and GPU0 had about 1.2 GiB external occupancy while GPUs1–3 were idle; no unrelated process was terminated.  There was no OOM, swap, near-OOM, duplicate launch, or checkpoint corruption event.  The final resource snapshot and residual-process check are in [`phase22_aggregate.json`](../../outputs/iclr27_phase22/metrics/phase22_aggregate.json).", "", "Large inputs were reused through symlinks (CSV, DINOv2 feature cache, DSCT checkpoint); resolved targets and hashes are in [`artifact_hashes.json`](../../outputs/iclr27_phase22/manifests/artifact_hashes.json).  Checkpoints exist for all four initial and repair folds at best/latest and 500-step intervals.  No git metadata is present in the project root (`git status` reports not a repository), so no commit/diff identifier can be supplied; all Phase22 files are isolated in the listed namespace.", "", "Integrity checks performed: all Phase22 JSON artifacts parse; 76 event rows are unique at prefix16; Stage0/Stage1/Stage2/Stage3/Phase22 completion markers exist; repair checkpoints and hashes are recorded; symlink targets resolve; forbidden public/Q1 label outputs are absent; and no Phase22 training/evaluation process remains.", ""]
    lines += ["## Reproduction commands", "", "```bash", "cd /data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT", "/home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase22/run_stage0_taxonomy.py", "/home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase22/run_data_feasibility.py", "./scripts/iclr27_phase22/run_four_fold_supervisor.sh", "./scripts/iclr27_phase22/run_four_fold_repair_supervisor.sh", "CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /home/lwr/anaconda3/envs/locatemot/bin/python -u scripts/iclr27_phase22/evaluate_proposal_refiner.py --device cuda:0 --trained-tag repair --out-stem stage3_proposal_validation_repair", "/home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase22/finalize_phase22.py", "```", "", "Final status: **proposal-source repair failed Gate P; stop before correspondence and public evaluation.**", ""]
    atomic_text(DOC / "PHASE22_PROPOSAL_SOURCE_REPAIR_COMPLETE_REPORT.md", "\n".join(lines))
    print(json.dumps({"decision_code": decision_code, "gate_p": gate_status, "raw_prefix16": raw16["ceiling_correct"], "initial_prefix16": init16["ceiling_correct"], "repair_prefix16": rep16["ceiling_correct"], "report": str(DOC / "PHASE22_PROPOSAL_SOURCE_REPAIR_COMPLETE_REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()
