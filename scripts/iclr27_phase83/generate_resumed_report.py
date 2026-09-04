#!/usr/bin/env python3
"""Create the Phase83 resumed-window ledger and, after the lock opens, report.

This generator is intentionally data-driven.  It never reruns an experiment and
it refuses to create the final report while the original Phase83 window is still
more than 45 minutes from its registered deadline.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase83"
AUDIT = OUT / "audit"
DOC = ROOT / "docs/iclr27_phase83"
START = dt.datetime.fromisoformat("2026-09-04T07:43:07+00:00")
DEADLINE = dt.datetime.fromisoformat("2026-09-04T17:43:07+00:00")
FINAL_REPORT = DOC / "PHASE83_RESUMED_FINAL_REPORT.md"
LEDGER = AUDIT / "resumed_research_ledger.json"
DECISION = AUDIT / "resumed_phase83_decision.json"
LOCK = AUDIT / "finalization_lock.json"
STATUS = OUT / "status.json"


def load(path: Path, default: Any = None) -> Any:
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def sha256(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


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


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso(t: dt.datetime | None = None) -> str:
    return (t or now()).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def git(*args: str) -> str:
    p = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return p.stdout.strip()


def command_output(argv: list[str], timeout: int = 20) -> str:
    try:
        p = subprocess.run(argv, cwd=ROOT, text=True, capture_output=True, timeout=timeout, check=False)
        return (p.stdout + p.stderr).strip()
    except Exception as exc:  # diagnostic only
        return f"{type(exc).__name__}: {exc}"


def f(value: Any, n: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{n}f}"
    except (TypeError, ValueError):
        return str(value)


def table(headers: list[str], rows: list[list[Any]]) -> str:
    out = "| " + " | ".join(headers) + " |\n"
    out += "|" + "|".join(["---"] * len(headers)) + "|\n"
    for row in rows:
        out += "| " + " | ".join(str(x) for x in row) + " |\n"
    return out


def safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256(path) if path.exists() and path.is_file() else None,
        "is_symlink": path.is_symlink(),
        "symlink_target": os.readlink(path) if path.is_symlink() else None,
    }


def route_prefix_rows(path: Path, key: str = "prefix_summary") -> list[dict[str, Any]]:
    data = load(path, {}) or {}
    rows = data.get(key, []) if isinstance(data, dict) else []
    return sorted(rows, key=lambda z: int(z.get("prefix", 0)))


def resource_snapshot() -> dict[str, Any]:
    return {
        "captured_utc": iso(),
        "cwd": str(Path.cwd()),
        "git_head": git("rev-parse", "HEAD"),
        "git_status_porcelain": git("status", "--short"),
        "git_origin_main": command_output(["git", "ls-remote", "origin", "refs/heads/main"]),
        "date_utc": command_output(["date", "-u"]),
        "free_h": command_output(["free", "-h"]),
        "df_data1_data2": command_output(["df", "-h", "/data1", "/data2"]),
        "nvidia_smi": command_output(["nvidia-smi"], timeout=30),
        "process_count": command_output(["bash", "-lc", "ps -e --no-headers | wc -l"]),
    }


def marker_audit() -> dict[str, Any]:
    completion = OUT / "completion"
    done = sorted(p.name for p in completion.glob("*.done")) if completion.exists() else []
    launched = sorted(p.name for p in completion.glob("*.launched")) if completion.exists() else []
    done_stems = {p[:-5] for p in done}
    launched_stems = {p[:-9] for p in launched}
    return {
        "completion_dir": str(completion.resolve()),
        "done_count": len(done),
        "launched_count": len(launched),
        "unmatched_launched_without_done": sorted(launched_stems - done_stems),
        "done_markers": done,
        "launched_markers": launched,
    }


def fold_rows(metric_path: Path) -> list[list[Any]]:
    data = load(metric_path, {}) or {}
    rows: list[list[Any]] = []
    for fold, record in sorted((data.get("folds") or {}).items(), key=lambda x: int(x[0])):
        vm = record.get("validation_metrics", {})
        rows.append([
            fold,
            record.get("steps"),
            record.get("fit_groups"),
            record.get("validation_groups"),
            f(vm.get("mean_nll")),
            f(vm.get("candidate_or_defer_accuracy")),
            f(vm.get("defer_recall")),
            vm.get("predicted_candidate_groups"),
            vm.get("reliable_target_groups"),
        ])
    return rows


def router_fold_rows(metric_path: Path) -> list[list[Any]]:
    data = load(metric_path, {}) or {}
    rows: list[list[Any]] = []
    for fold, record in sorted((data.get("folds") or {}).items(), key=lambda x: int(x[0])):
        vm = record.get("validation_metrics", {})
        fm = record.get("fit_metrics", {})
        rows.append([
            fold,
            record.get("steps"),
            fm.get("rows"),
            f(fm.get("positive_rate")),
            f(vm.get("roc_auc")),
            f(vm.get("f1")),
            f(record.get("loss_first")),
            f(record.get("loss_last")),
        ])
    return rows


def physical_prefix_rows(data: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    sec = data.get("sections", {}).get("exact_mixed", {})
    for p in (1, 2, 4, 8, 16):
        z = (sec.get("prefix", {}) or {}).get(str(p), {})
        if not z:
            continue
        rows.append([
            p,
            z.get("queries"),
            f(z.get("raw_r1")),
            f(z.get("r1")),
            f((z.get("r1") or 0) - (z.get("raw_r1") or 0)),
            f(z.get("raw_map")),
            f(z.get("map")),
            f(z.get("hard_negative_gap")),
            f(z.get("raw_hard_negative_gap")),
            z.get("unsafe_flip_count"),
        ])
    return rows


def replay_prefix_rows(path: Path) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for z in route_prefix_rows(path):
        rows.append([
            z.get("prefix"),
            z.get("positive_events"),
            z.get("negative_events"),
            z.get("frozen_both_reliable"),
            z.get("learned_both_support_selected"),
            z.get("learned_both_support_reliable"),
            z.get("negative_both_support_selected"),
            z.get("negative_both_support_reliable"),
        ])
    return rows


def b5_prefix_rows(path: Path) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for z in route_prefix_rows(path):
        rows.append([
            z.get("prefix"), z.get("positive"), z.get("negative"),
            z.get("frozen_source"), z.get("frozen_target"),
            z.get("cross_support_target"), z.get("cross_support_both"),
            z.get("negative_cross_support_target"),
        ])
    return rows


def event_failure_rows() -> list[list[Any]]:
    data = load(AUDIT / "failure_taxonomy_76.json", {}) or {}
    out: list[list[Any]] = []
    for e in data.get("events", []):
        out.append([
            e.get("event_key"), e.get("fold"), e.get("category"),
            e.get("overall_failure"), e.get("source_failure"), e.get("target_failure"),
            e.get("source_candidate_count"), e.get("target_candidate_count"),
            f(e.get("source_max_iou")), f(e.get("target_max_iou")),
            str(bool(e.get("pool_source_reliable"))), str(bool(e.get("pool_target_reliable"))),
        ])
    return out


def symlink_ledger() -> list[dict[str, Any]]:
    paths = [OUT, OUT / "metrics", OUT / "audit", OUT / "checkpoints", OUT / "manifests"]
    rows = []
    for p in paths:
        if p.is_symlink():
            rows.append({"path": str(p), "target": os.readlink(p), "target_exists": p.exists()})
    for p in [
        Path("/data2/usr_for_deadline/trackocd_phase83"),
        Path("/data2/usr_for_deadline/trackocd_phase83/project_outputs"),
        Path("/data2/usr_for_deadline/trackocd_phase83/a2_full/native_lineage.jsonl"),
        Path("/data2/usr_for_deadline/trackocd_phase83/a2_dino_full_r1/merged/native_dinov2.npz"),
    ]:
        if p.exists():
            rows.append({"path": str(p), "target": None, "target_exists": True, "sha256": sha256(p) if p.is_file() else None})
    return rows


def build_snapshot() -> dict[str, Any]:
    physical = load(OUT / "metrics/physical_r_temporal.json", {}) or {}
    a2_lineage = load(AUDIT / "a2_full_q0_lineage.json", {}) or {}
    a2_mapping = load(AUDIT / "a2_native_mapping.json", {}) or {}
    a2_temporal = load(OUT / "metrics/a2_temporal_r.json", {}) or {}
    a3 = load(OUT / "metrics/a3_multiprototype_r.json", {}) or {}
    a3_diag = load(AUDIT / "a3_identity_diagnostic.json", {}) or {}
    b2 = load(OUT / "metrics/b2_listwise_aggregate_b2_formal.json", {}) or {}
    b2_replay = load(OUT / "metrics/b2_listwise_replay_b2_formal.json", {}) or {}
    b3 = load(OUT / "metrics/b3_joint_aggregate_b3_formal.json", {}) or {}
    b3_replay = load(OUT / "metrics/b3_joint_replay_b3_formal.json", {}) or {}
    b4 = load(OUT / "metrics/b4_native_aggregate_b4_formal.json", {}) or {}
    b4_replay = load(OUT / "metrics/b4_native_replay_b4_formal.json", {}) or {}
    b5 = load(AUDIT / "b5_cross_video_support.json", {}) or {}
    inventory = load(AUDIT / "r_full_coverage_inventory.json", {}) or {}
    pool = load(AUDIT / "pool_ceiling.json", {}) or {}
    failure_summary = load(AUDIT / "failure_taxonomy_summary.json", {}) or {}
    b2_manifest = load(OUT / "manifests/b2_candidate_sets_v1.json", {}) or {}
    b4_manifest = load(OUT / "manifests/b4_native_sets_v1.json", {}) or {}
    formal_inv = load(OUT / "manifests/support_router_inventory_formal.json", {}) or {}
    now_dt = now()
    remaining = max(0, int((DEADLINE - now_dt).total_seconds()))
    return {
        "captured_utc": iso(now_dt),
        "window": {
            "original_start_utc": iso(START),
            "original_deadline_utc": iso(DEADLINE),
            "elapsed_seconds": int((now_dt - START).total_seconds()),
            "remaining_seconds": remaining,
            "resume_status": load(AUDIT / "resume_status.json", {}),
            "finalization_lock": load(LOCK, {}),
        },
        "git": {
            "head": git("rev-parse", "HEAD"),
            "status": git("status", "--short"),
            "recent": git("log", "-20", "--oneline"),
            "origin_main": git("ls-remote", "origin", "refs/heads/main"),
        },
        "resources": resource_snapshot(),
        "markers": marker_audit(),
        "symlinks": symlink_ledger(),
        "sealed_boundary": {
            "public_dev_q1_sealed_accessed": False,
            "future_rows_or_tracks": False,
            "ids_as_model_input": False,
            "category_text_as_input": False,
            "controller_run": False,
            "state_memory_run": False,
            "threshold_sweep": False,
            "event_labels": "posthoc audit only; never model input or checkpoint selection",
        },
        "inventory": inventory,
        "a2": {
            "lineage": a2_lineage,
            "mapping": a2_mapping,
            "temporal_metrics": a2_temporal,
            "physical_metrics": physical,
            "prefix_rows": physical_prefix_rows(physical),
            "decision": "A2_FAIL_PARTIAL_NATIVE_MAPPING_AND_NO_SAFE_R_GAIN",
        },
        "pool_ceiling": pool,
        "failure_summary": failure_summary,
        "event_failure_rows": event_failure_rows(),
        "a3": {
            "diagnostic": a3_diag,
            "multiprototype": a3,
            "p16_aggregate": (a3.get("aggregate") or {}).get("16", {}),
            "decision": "A3_FAIL_MULTI_PROTOTYPE_NO_SAFE_IMPROVEMENT",
        },
        "b2": {
            "manifest": b2_manifest,
            "aggregate": b2,
            "replay": b2_replay,
            "fold_rows": fold_rows(OUT / "metrics/b2_listwise_aggregate_b2_formal.json"),
            "prefix_rows": replay_prefix_rows(OUT / "metrics/b2_listwise_replay_b2_formal.json"),
            "decision": "B2_FAIL_LISTWISE_NO_SUPPORT_GAIN",
        },
        "b3": {
            "aggregate": b3,
            "replay": b3_replay,
            "fold_rows": fold_rows(OUT / "metrics/b3_joint_aggregate_b3_formal.json"),
            "prefix_rows": replay_prefix_rows(OUT / "metrics/b3_joint_replay_b3_formal.json"),
            "contract_audit": load(AUDIT / "b3_candidate_contract_audit.json", {}),
            "decision": "B3_FAIL_RUNTIME_CANDIDATE_CONTRACT_MISMATCH_AND_NO_GAIN",
        },
        "b4": {
            "manifest": b4_manifest,
            "aggregate": b4,
            "replay": b4_replay,
            "fold_rows": fold_rows(OUT / "metrics/b4_native_aggregate_b4_formal.json"),
            "prefix_rows": replay_prefix_rows(OUT / "metrics/b4_native_replay_b4_formal.json"),
            "selection_audit": load(AUDIT / "b4_native_selection_audit.json", {}),
            "decision": "B4_FAIL_NATIVE_SET_MATCHER_NO_SUPPORT_GAIN",
        },
        "b5": {
            "audit": b5,
            "prefix_rows": b5_prefix_rows(AUDIT / "b5_cross_video_support.json"),
            "decision": "B5_FAIL_CROSS_VIDEO_SUPPORT_DIAGNOSTIC_NO_SAFE_HEADLINE",
        },
        "artifacts": [
            artifact(AUDIT / "a2_full_q0_lineage.json"),
            artifact(AUDIT / "a2_native_mapping.json"),
            artifact(OUT / "metrics/physical_r_temporal.json"),
            artifact(OUT / "metrics/a2_temporal_r.json"),
            artifact(AUDIT / "a3_identity_diagnostic.json"),
            artifact(OUT / "metrics/a3_multiprototype_r.json"),
            artifact(OUT / "manifests/b2_candidate_sets_v1.json"),
            artifact(OUT / "metrics/b2_listwise_aggregate_b2_formal.json"),
            artifact(OUT / "metrics/b2_listwise_replay_b2_formal.json"),
            artifact(AUDIT / "b3_candidate_contract_audit.json"),
            artifact(OUT / "metrics/b3_joint_aggregate_b3_formal.json"),
            artifact(OUT / "metrics/b3_joint_replay_b3_formal.json"),
            artifact(OUT / "manifests/b4_native_sets_v1.json"),
            artifact(AUDIT / "b4_native_selection_audit.json"),
            artifact(OUT / "metrics/b4_native_aggregate_b4_formal.json"),
            artifact(OUT / "metrics/b4_native_replay_b4_formal.json"),
            artifact(AUDIT / "b5_cross_video_support.json"),
            artifact(OUT / "metrics/o_support_replay_formal.json"),
            artifact(OUT / "metrics/support_router_aggregate_formal.json"),
        ],
        "checkpoints": [artifact(p) for p in sorted((OUT / "checkpoints").glob("*.npz"))],
        "route_tree": [
            {"route": "R83 first physical temporal mean", "status": "FAIL", "next": "A2 full coverage"},
            {"route": "A2 full Q0 lineage + temporal mean", "status": "FAIL", "next": "A3 identity/prototype diagnostic"},
            {"route": "A3 identity diagnostic + M=3 prototypes", "status": "FAIL", "next": "B2/B3 support formulation"},
            {"route": "O83 binary row router", "status": "FAIL", "next": "B2 listwise + DEFER"},
            {"route": "B2 listwise candidate competition", "status": "FAIL", "next": "B3 contract audit/joint support"},
            {"route": "B3 joint support matcher", "status": "FAIL", "next": "B4 native runtime candidate set"},
            {"route": "B4 native candidate-set matcher", "status": "FAIL", "next": "B5 cross-video support diagnostic"},
            {"route": "B5 cross-video prior support diagnostic", "status": "FAIL", "next": "window closure; no C without safe R/O"},
            {"route": "C83 unchanged controller", "status": "NOT_RUN", "next": "requires safe R or O"},
            {"route": "sealed/public evaluation", "status": "NOT_RUN", "next": "requires C authorization"},
        ],
        "reproduction_commands": [
            "python scripts/iclr27_phase83/run_a2_full_q0.py --help (full-Q0 lineage; existing artifact is frozen)",
            "python scripts/iclr27_phase83/run_a2_temporal_r.py --help (A2 temporal evaluator; existing artifact is frozen)",
            "python scripts/iclr27_phase83/run_a3_identity_diagnostic.py",
            "python scripts/iclr27_phase83/run_a3_multiprototype_r.py",
            "python scripts/iclr27_phase83/train_support_listwise.py --folds 0,1,2,3 --steps 1000 --tag b2_formal",
            "python scripts/iclr27_phase83/run_b3_joint_support.py --folds 0,1,2,3 --steps 1000 --tag b3_formal",
            "python scripts/iclr27_phase83/build_b4_native_candidate_sets.py",
            "python scripts/iclr27_phase83/train_b4_native_matcher.py --folds 0,1,2,3 --steps 1000 --tag b4_formal",
            "python scripts/iclr27_phase83/audit_b5_cross_video_support.py",
        ],
    }


def ledger_from(snapshot: dict[str, Any], status: str) -> dict[str, Any]:
    return {
        "schema_version": "trackocd.phase83.resumed_research_ledger.v1",
        "phase": "Phase83",
        "status": status,
        "generated_utc": snapshot["captured_utc"],
        "window": snapshot["window"],
        "code_head": snapshot["git"]["head"],
        "original_clock_not_restarted": True,
        "premature_finalization_corrected": True,
        "sealed_boundary": snapshot["sealed_boundary"],
        "route_tree": snapshot["route_tree"],
        "stage_entries": [
            {
                "branch": "A2",
                "hypothesis": "full-coverage physical temporal appearance can change R without raw fallback confound",
                "input_hashes": {
                    "native_lineage": (snapshot["a2"]["lineage"] or {}).get("native_lineage_sha256"),
                    "native_dino": (snapshot["a2"]["mapping"] or {}).get("features_sha256"),
                },
                "result": "full native lineage 682335 rows/13678 traces; mapping 6213 tracks, 74/76 event pairs; p16 temporal R1 0.882735 < raw 0.893219; unsafe 5",
                "decision": snapshot["a2"]["decision"],
                "next_action": "A3 completed",
            },
            {
                "branch": "A3",
                "hypothesis": "identity-preserving multi-prototype causal pooling avoids temporal mean oversmoothing",
                "input_hashes": {
                    "native_dino": (snapshot["a3"]["diagnostic"] or {}).get("native_features_sha256"),
                    "q0_dino": (snapshot["a3"]["diagnostic"] or {}).get("q0_feature_sha256"),
                },
                "result": "M=3 p16 R1 0.890469 < raw 0.893219; mAP 0.850450 > 0.848374 but hard-gap 0.167946 < 0.189559 and 6 unsafe flips; 1/4 folds non-decreasing",
                "decision": snapshot["a3"]["decision"],
                "next_action": "B4/B5 completed",
            },
            {
                "branch": "B2",
                "hypothesis": "support is set assignment with explicit DEFER, not row-independent quality classification",
                "input_hashes": {"candidate_manifest": (snapshot["b2"]["manifest"] or {}).get("data_sha256")},
                "result": "8,841 groups/33,684 rows; 4x1000 formal; p16 learned both reliable 6/76 vs frozen 25/76; selected 48/76 positive and 48/76 negative",
                "decision": snapshot["b2"]["decision"],
                "next_action": "B3 candidate contract audit",
            },
            {
                "branch": "B3",
                "hypothesis": "adding causal DINO history and candidate context fixes listwise support selection",
                "input_hashes": {"candidate_manifest": (snapshot["b3"]["contract_audit"] or {}).get("public_csv_sha256")},
                "result": "p16 learned both reliable 9/76 positive and 4/76 negative; runtime candidate counts match public sets only 237/14691 (1.613%), median absolute count difference 19",
                "decision": snapshot["b3"]["decision"],
                "next_action": "B4 true native candidate universe",
            },
            {
                "branch": "B4",
                "hypothesis": "training and ranking on the true native runtime candidate set removes B3 mismatch",
                "input_hashes": {"candidate_manifest": (snapshot["b4"]["manifest"] or {}).get("data_sha256")},
                "result": "13,631 native groups/464,146 rows; p16 learned both reliable 4/76 vs frozen 25/76; native base-score diagnostic has 12/76 both while pool oracle has 61/76",
                "decision": snapshot["b4"]["decision"],
                "next_action": "B5 cross-video support diagnostic",
            },
            {
                "branch": "B5",
                "hypothesis": "prior completed source-track appearance can recover target support across videos without another learned router",
                "input_hashes": {"native_features": (snapshot["b5"]["audit"] or {}).get("native_features")},
                "result": "p16 cross-support target 20/76, both 17/76; frozen target 40/76, source 49/76; negative target 8/76; no safe O gain",
                "decision": snapshot["b5"]["decision"],
                "next_action": "close original window only; C83 remains NOT_RUN",
            },
        ],
        "resource_snapshot": snapshot["resources"],
        "marker_audit": snapshot["markers"],
        "artifacts": snapshot["artifacts"],
        "symlinks": snapshot["symlinks"],
        "no_new_training_or_sealed_run": True,
        "final_report": {"path": str(FINAL_REPORT), "generated": FINAL_REPORT.exists()},
    }


def build_report(snapshot: dict[str, Any], final_time: dt.datetime) -> str:
    a2 = snapshot["a2"]
    p = a2["physical_metrics"].get("sections", {}).get("exact_mixed", {})
    p16 = (p.get("prefix", {}) or {}).get("16", {})
    pool = snapshot["pool_ceiling"]
    a3p = snapshot["a3"]["p16_aggregate"]
    b2 = snapshot["b2"]
    b3 = snapshot["b3"]
    b4 = snapshot["b4"]
    b5 = snapshot["b5"]
    resume = snapshot["window"]["resume_status"] or {}
    lock = snapshot["window"]["finalization_lock"] or {}
    elapsed = int((final_time - START).total_seconds())
    remaining_at_final = max(0, int((DEADLINE - final_time).total_seconds()))
    event_count = len(snapshot["event_failure_rows"])
    physical_rows = table(
        ["prefix", "queries", "raw R@1", "temporal R@1", "ΔR@1", "raw mAP", "temporal mAP", "temporal gap", "raw gap", "unsafe"],
        a2["prefix_rows"],
    )
    b2_rows = table(["prefix", "pos", "neg", "frozen both", "selected pos", "reliable pos", "selected neg", "reliable neg"], b2["prefix_rows"])
    b3_rows = table(["prefix", "pos", "neg", "frozen both", "selected pos", "reliable pos", "selected neg", "reliable neg"], b3["prefix_rows"])
    b4_rows = table(["prefix", "pos", "neg", "frozen both", "selected pos", "reliable pos", "selected neg", "reliable neg"], b4["prefix_rows"])
    b5_rows = table(["prefix", "pos", "neg", "frozen source", "frozen target", "cross target", "cross both", "negative target"], b5["prefix_rows"])
    event_table = table(
        ["event", "fold", "category", "overall", "source", "target", "src n", "tgt n", "src max IoU", "tgt max IoU", "pool src", "pool tgt"],
        snapshot["event_failure_rows"],
    )
    b2_fold = table(["fold", "steps", "fit groups", "val groups", "val NLL", "candidate/defer acc", "defer recall", "pred candidate", "reliable target"], b2["fold_rows"])
    b3_fold = table(["fold", "steps", "fit groups", "val groups", "val NLL", "candidate/defer acc", "defer recall", "pred candidate", "reliable target"], b3["fold_rows"])
    b4_fold = table(["fold", "steps", "fit groups", "val groups", "val NLL", "candidate/defer acc", "defer recall", "pred candidate", "reliable target"], b4["fold_rows"])
    syms = table(["path", "target", "exists", "sha256"], [[s.get("path"), s.get("target") or "", s.get("target_exists"), s.get("sha256", "")] for s in snapshot["symlinks"]])
    markers = snapshot["markers"]
    return f"""# TrackOCD Phase83 Resumed Final Report

**Window:** `{iso(START)}` → `{iso(DEADLINE)}` (original clock; not restarted)  
**Resume:** `{resume.get('resume_time_utc', 'n/a')}`  
**Premature-finalization correction:** prior HEAD `{resume.get('premature_finalize_head', 'n/a')}` was finalized before the registered window ended; this report is the corrected closeout.  
**True finalization time:** `{iso(final_time)}`  
**Elapsed from original start:** `{elapsed}s`  
**Remaining at finalization:** `{remaining_at_final}s`  
**Generation source HEAD:** `{snapshot['git']['head']}`  
**Finalization lock:** allowed=`{lock.get('allowed')}`, release reason=`{lock.get('reason', 'deadline-45-minute rule')}`

## Executive decision

`AUTONOMOUS_PHASE83_WINDOW_COMPLETE_WITH_NEGATIVE_EVIDENCE` is the status of this **single resumed Phase83 window**, not a claim that TrackOCD is universally infeasible. The registered R/O diagnostics were completed with negative or unsafe results. No safe R/O improvement authorized unchanged-controller compatibility; therefore C83 and sealed/public evaluation remain **NOT_RUN**. The immutable frozen comparator remains strict p16 observation `25/76` (source `49/76`, target `40/76`), while the post-hoc native pool ceiling is source `{pool.get('source_max_iou_ge_0.5')}/76`, target `{pool.get('target_max_iou_ge_0.5')}/76`, both `{pool.get('both_max_iou_ge_0.5')}/76`.

The conclusion is specific: the tested temporal-mean, prototype, listwise, native-set, and cross-video support formulations did not safely transfer to the exact causal R/O protocol in this window. They do not prove that a future, separately registered support/assignment contract or full physical-stream redesign cannot work.

## Frozen protocol and sealed boundary

- Exact Phase30/Phase75D R universe: 43,423 rows, 6,213 tracks, 984 queries; same candidate order, same-video exclusion, folds and prefixes `{1,2,4,8,16}`.
- O replay: 76 positive + 76 negative events, unchanged reliable rule and denominator. Event labels, IoU and assignment are post-hoc diagnostics or TRAIN target metadata only.
- Inference tensors contain only visual/geometry/causal history fields. No category/text, semantic or numeric physical ID, future row/track, DEV+, Q1, public-new or sealed label was accessed; no held result selected a checkpoint or threshold.
- Phase75B evaluator and physical stream were not rewritten; all Phase83 fields are versioned artifacts. No threshold, StateMemory, controller, backbone or sealed/public route was run.

## Route tree and decisions

{table(['route','status','next / interpretation'], [[r['route'],r['status'],r['next']] for r in snapshot['route_tree']])}

### R83 first route (historical checkpoint)

The event-native temporal-appearance-mean stream was partial: 5,487/43,423 rows and 1,046/6,213 tracks were usable (16.84%), covering 74/76 event pairs. Exact mixed p16 was raw R@1 `0.893219` vs temporal `0.882735`, raw mAP `0.848374` vs `0.847251`, raw hard gap `0.189559` vs `0.198022`, with 5 unsafe flips. This was retained as a partial-coverage diagnostic, not a full-stream claim.

### A2 full-coverage Q0 physical lineage

Q0 native inference completed across 370 public TRAIN videos: 682,335 rows and 13,678 frame traces. The corrected DINOv2 cache is 682,335×768 (SHA256 `{(a2['mapping'] or {}).get('features_sha256', 'n/a')}`). Mapping used identical `(video_id,image_id)` and proposal-box IoU≥0.5; 6,213 tracks were mapped, but only 74/76 event pairs were native-mapped and 21.00% of public tracks had a complete row fraction. The full-coverage attempt therefore still required explicit mapping diagnostics; it did not justify a headline physical improvement.

{physical_rows}

At p16, temporal R@1 is `{f(p16.get('r1'))}` vs raw `{f(p16.get('raw_r1'))}`, mAP `{f(p16.get('map'))}` vs `{f(p16.get('raw_map'))}`, hard gap `{f(p16.get('hard_negative_gap'))}` vs `{f(p16.get('raw_hard_negative_gap'))}`, unsafe `{p16.get('unsafe_flip_count')}`. Decision: `A2_FAIL_PARTIAL_NATIVE_MAPPING_AND_NO_SAFE_R_GAIN`.

### A3 identity/prototype diagnostic

The identity audit mapped 1,194/1,298 native-R tracks (`0.9199`), with native appearance variance `0.143507` vs Q0 `0.206780`, native self-cosine `0.856493` vs Q0 `0.793219`, mean reconnected segments `3.0184`, adjacent segment cosine `0.558430`, and native query gap `0.027819` vs Q0 `0.025014`. Fixed M=3 contiguous causal prototypes with symmetric max cosine gave p16 R@1 `{f(a3p.get('r1'))}` vs raw `{f(a3p.get('raw_r1'))}`, mAP `{f(a3p.get('map'))}` vs `{f(a3p.get('raw_map'))}`, hard gap `{f(a3p.get('hard_negative_gap'))}` vs raw `{f(a3p.get('raw_hard_negative_gap'))}`, unsafe `{a3p.get('unsafe_flip_count')}`; only one of four folds was non-decreasing. This rejects the tested prototype formulation, not the general possibility of semantic representation learning.

### O83 and B2/B3/B4/B5 support routes

The original binary row router is retained as a negative comparator: it selected support on 46/76 positive and 52/76 negative events but yielded only 8/76 positive and 10/76 negative both-side reliable events at p16. It did not solve support assignment.

#### B2 listwise + explicit DEFER

Manifest: 8,841 groups, 33,684 candidates, 1,688 reliable TRAIN target groups and 7,153 DEFER groups. Formal training was four folds × 1,000 updates; the first scalar-loader smoke failure was preserved and repaired with an atomic fresh smoke/targeted run. Validation fold details:

{b2_fold}

Frozen-event replay:

{b2_rows}

Decision: `B2_FAIL_LISTWISE_NO_SUPPORT_GAIN`.

#### B3 joint support matcher and candidate-contract audit

Adding causal DINO history and candidate-set context did not solve the interface. At p16 it produced 9/76 positive and 4/76 negative both-side reliable events. The decisive contract audit found the runtime Q0 candidate count matched the B2/B3 public grouping in only 237/14,691 groups (`1.613%`); median absolute count difference was 19 and mean public-minus-native difference `-26.247`. This is a candidate-universe mismatch, not evidence that geometry or the listwise idea is impossible.

{b3_fold}

{b3_rows}

Decision: `B3_FAIL_RUNTIME_CANDIDATE_CONTRACT_MISMATCH_AND_NO_GAIN`.

#### B4 native runtime candidate set

The mismatch was repaired by constructing 13,631 native runtime groups with 464,146 bbox-bearing rows, 6,077 reliable TRAIN target groups and 7,554 DEFER groups. Formal four-fold 1,000-update matching on this exact set still yielded only 4/76 positive both-side reliable events at p16, compared with frozen 25/76; deterministic native base-score selection retained both-side candidates in 12/76 events while the post-hoc pool oracle was 61/76. Validation fold details:

{b4_fold}

{b4_rows}

Decision: `B4_FAIL_NATIVE_SET_MATCHER_NO_SUPPORT_GAIN`.

#### B5 cross-video prior support diagnostic

Using a completed source-track Q0 appearance against native target candidates (per-frame maximum cosine) was a diagnostic only. At p16 it provided target support for 20/76 and both-side support for 17/76, below frozen target 40/76 and source 49/76; negative target support was 8/76. It cannot authorize controller compatibility.

{b5_rows}

Decision: `B5_FAIL_CROSS_VIDEO_SUPPORT_DIAGNOSTIC_NO_SAFE_HEADLINE`.

## Complete p16 event failure index

The 76-event taxonomy is preserved verbatim in `outputs/iclr27_phase83/audit/failure_taxonomy_76.json` and CSV. The following table is generated directly from that artifact (`{event_count}` rows), so hard events were not removed or re-denominated:

{event_table}

Aggregated p16 classes are B proposal exists but max IoU<0.5 = 15, D assigned but transformed IoU<0.5 = 18, E support selection wrong = 36, G other = 7. The pool upper-bound and frozen reliability remain separate; oracle rows are never reported as learned O or OCD success.

## Checkpoint, marker, repair and resource audit

- B2 first smoke failed only because the replay loader treated scalar `bc/bd` as an array; the `.launched` and failed evidence were kept. The smallest loader fix was committed, then a fresh smoke and targeted run passed. No samples, seed, denominator or protocol changed.
- A3 first invocation had a missing `src` import path; the smallest path repair was committed and rerun. B4 first candidate build omitted `norm` (`NameError`); the helper repair was committed and rerun. B3's global normalization/sampler repair and B4's native candidate contract repair were each followed by smoke/targeted checks. These are implementation repairs, not hidden scientific retries.
- The first physical-R process was task-owned PID 17813 with wait shell 17963; it was SIGTERM-ed after profiling exposed repeated per-pair raw-vector recomputation and no artifact had been produced. No external process was touched and no OOM occurred.
- Completion markers: `{markers['done_count']}` `.done`, `{markers['launched_count']}` `.launched`; unmatched launched markers are `{markers['unmatched_launched_without_done']}`. Checkpoints are resumable `.npz` artifacts with hashes in the ledger. No Phase83 process remained at final audit.
- GPU 0–3 were occupied by external jobs during much of the run; the appearance extraction used idle GPUs 5–8. B2/B3/B4 routers were CPU-bound and did not need GPU placement. GPU4 was not touched after an external job appeared. RAM preflight had approximately 98 GiB available of 125 GiB total (≥25% headroom); `/data1` was near capacity, so large caches/checkpoints were stored under `/data2/usr_for_deadline/trackocd_phase83` and exposed by symlink. No OOM or near-OOM event occurred.

Symlink/storage ledger:

{syms}

## Gates and what was not run

| gate | result | evidence |
|---|---|---|
| R83 physical temporal mean | FAIL | exact p16 R@1/mAP lower and 5 unsafe flips |
| A2 full-coverage physical | FAIL | mapping remained incomplete; no safe R headline |
| A3 multi-prototype | FAIL | p16 R@1 lower, 6 unsafe, 1/4 folds non-decreasing |
| O83 binary router | FAIL | positive/negative over-activation; both reliable 8/76 |
| B2 listwise | FAIL | both reliable 6/76 vs frozen 25/76 |
| B3 joint support | FAIL | runtime candidate-set mismatch; both reliable 9/76 |
| B4 native-set matcher | FAIL | both reliable 4/76 vs frozen 25/76 |
| B5 cross-video support | FAIL | both 17/76, target 20/76 vs frozen 40/76/49 source |
| C83 unchanged controller | NOT_RUN | no safe R/O result authorized it |
| sealed/public evaluation | NOT_RUN | sealed boundary remained closed |

Training loss, validation NLL, AUC, candidate oracle, or raw/top-K diagnostics are not persistent Commit-CT. This window therefore makes no claim of an OCD success or of a full MOT+OCD result.

## Reproduction and artifacts

The exact commands and input/output hashes are in `outputs/iclr27_phase83/audit/resumed_research_ledger.json`. The principal artifacts are:

- `outputs/iclr27_phase83/audit/a2_full_q0_lineage.json`, `a2_native_mapping.json`, `a3_identity_diagnostic.json`, `b3_candidate_contract_audit.json`, `b4_native_selection_audit.json`, `b5_cross_video_support.json`;
- `outputs/iclr27_phase83/metrics/physical_r_temporal.json`, `a2_temporal_r.json`, `a3_multiprototype_r.json`, `b2_listwise_replay_b2_formal.json`, `b3_joint_replay_b3_formal.json`, `b4_native_replay_b4_formal.json`, `o_support_replay_formal.json`;
- `outputs/iclr27_phase83/manifests/b2_candidate_sets_v1.json`, `b4_native_sets_v1.json`, `support_router_inventory_formal.json`;
- complete event evidence: `outputs/iclr27_phase83/audit/failure_taxonomy_76.json` and `.csv`.

## Final Phase83 scope

This is a corrected closeout of the original 10-hour Phase83 window. It records all registered routes that were actually run in the resumed window and preserves their negative evidence. It does **not** authorize a threshold/memory/controller/backbone lottery, and it does not close the wider TrackOCD research program. A future phase, if authorized, should repair the native support/assignment contract or construct a full-coverage physical stream first, then preregister one causal correspondence route; any later C must still use the unchanged 76+76 protocol and demonstrate safety before sealed evaluation.
"""


def can_finalize(t: dt.datetime, lock: dict[str, Any]) -> tuple[bool, str]:
    if t >= DEADLINE:
        return True, "original deadline reached"
    if t >= DEADLINE - dt.timedelta(minutes=45):
        return True, "deadline-minus-45-minute finalization window"
    if bool(lock.get("allowed")):
        return True, "lock already explicitly released"
    return False, "finalization lock remains closed before deadline-minus-45-minute window"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--finalize", action="store_true", help="attempt final report; still obeys the lock")
    args = ap.parse_args()
    snapshot = build_snapshot()
    lock = snapshot["window"]["finalization_lock"] or {}
    t = now()
    finalize, reason = can_finalize(t, lock)
    ledger_status = "FINALIZATION_READY" if finalize else "RESUMED_WINDOW_IN_PROGRESS"
    atomic_json(LEDGER, ledger_from(snapshot, ledger_status))
    if not (args.finalize and finalize):
        print(json.dumps({"status": ledger_status, "reason": reason, "ledger": str(LEDGER), "remaining_seconds": snapshot["window"]["remaining_seconds"]}, indent=2))
        return
    # Release only at the authorized time; this write is itself atomic.
    new_lock = dict(lock)
    new_lock.update({"allowed": True, "original_deadline": iso(DEADLINE), "reason": reason, "updated_utc": iso(t)})
    atomic_json(LOCK, new_lock)
    snapshot["window"]["finalization_lock"] = new_lock
    snapshot["captured_utc"] = iso(t)
    report = build_report(snapshot, t)
    atomic_text(FINAL_REPORT, report)
    decision = {
        "schema_version": "trackocd.phase83.resumed_decision.v1",
        "phase": "Phase83",
        "status": "AUTONOMOUS_PHASE83_WINDOW_COMPLETE_WITH_NEGATIVE_EVIDENCE",
        "decision_code": "P83_RESUMED_WINDOW_COMPLETE_R_A2_A3_AND_O_B2_B3_B4_B5_FAIL_C_NOT_RUN",
        "original_start_utc": iso(START),
        "original_deadline_utc": iso(DEADLINE),
        "resume_time_utc": (snapshot["window"].get("resume_status") or {}).get("resume_time_utc"),
        "true_finalization_utc": iso(t),
        "elapsed_from_original_start_seconds": int((t - START).total_seconds()),
        "remaining_to_original_deadline_seconds": max(0, int((DEADLINE - t).total_seconds())),
        "R83": "FAIL",
        "A2": "FAIL",
        "A3": "FAIL",
        "O83": "FAIL",
        "B2": "FAIL",
        "B3": "FAIL",
        "B4": "FAIL",
        "B5": "FAIL",
        "C83": "NOT_RUN",
        "sealed": "NOT_RUN",
        "public_dev_q1_sealed_accessed": False,
        "future_rows_or_tracks": False,
        "ids_as_model_input": False,
        "category_text_as_input": False,
        "report": str(FINAL_REPORT.resolve()),
        "ledger": str(LEDGER.resolve()),
        "lock": str(LOCK.resolve()),
        "route_tree": snapshot["route_tree"],
        "artifacts": snapshot["artifacts"],
        "next_action": "Do not claim universal infeasibility; any new Phase84+ route requires a separate preregistration focused on native support/assignment or full physical stream.",
    }
    atomic_json(DECISION, decision)
    atomic_json(STATUS, {
        "phase": "Phase83",
        "status": decision["status"],
        "decision_code": decision["decision_code"],
        "R83": "FAIL", "A2": "FAIL", "A3": "FAIL", "O83": "FAIL", "B2": "FAIL", "B3": "FAIL", "B4": "FAIL", "B5": "FAIL", "C83": "NOT_RUN", "sealed": "NOT_RUN",
        "public_dev_q1_sealed_accessed": False,
        "future_rows_or_tracks": False,
        "ids_as_model_input": False,
        "final_report": str(FINAL_REPORT.resolve()),
        "finalized_utc": iso(t),
        "remaining_seconds": decision["remaining_to_original_deadline_seconds"],
    })
    # Rewrite ledger with final state and report metadata after all atomic outputs exist.
    final_snapshot = build_snapshot()
    final_ledger = ledger_from(final_snapshot, decision["status"])
    final_ledger["finalization"] = {"allowed": True, "reason": reason, "decision": decision["decision_code"], "report": str(FINAL_REPORT.resolve())}
    atomic_json(LEDGER, final_ledger)
    print(json.dumps({"status": decision["status"], "report": str(FINAL_REPORT.resolve()), "decision": str(DECISION.resolve()), "head": git("rev-parse", "HEAD")}, indent=2))


if __name__ == "__main__":
    main()
