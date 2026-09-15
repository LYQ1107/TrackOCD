#!/usr/bin/env python3
"""Completion-based TrackOCD v2 stage supervisor.

The supervisor is intentionally bounded: one invocation advances available
CPU stages and records a resource wait instead of launching duplicate work.
``--loop`` is available for a long-lived deployment and sleeps internally;
the agent itself never polls a long-running worker.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
FEATURE_PYTHON = os.environ.get("TRACKOCD_FEATURE_PYTHON", "/home/lwr/anaconda3/envs/ovtr/bin/python")
if not Path(FEATURE_PYTHON).is_file():
    FEATURE_PYTHON = PYTHON
sys.path.insert(0, str(ROOT))

from src.trackocd_v2.io import OUTPUT_TARGET, atomic_json, ensure_output_layout  # noqa: E402
from src.trackocd_v2.protocol import STAGES  # noqa: E402


STATE_PATH = OUTPUT_TARGET / "audit/autonomous_state.json"


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_state() -> dict:
    if not STATE_PATH.exists():
        subprocess.run([PYTHON, str(ROOT / "scripts/trackocd_v2/bootstrap.py")], cwd=ROOT, check=True)
    return json.loads(STATE_PATH.read_text())


def write_state(state: dict) -> None:
    state["updated_utc"] = now()
    atomic_json(STATE_PATH, state)


def mark_done(state: dict, stage: str, artifact: Path) -> None:
    completed = list(state.get("completed_stages", []))
    if stage not in completed:
        completed.append(stage)
    state["completed_stages"] = completed
    state["artifacts"][stage] = str(artifact.resolve())
    state["pending_stages"] = [x for x in STAGES if x not in completed]
    state["state"] = state["pending_stages"][0] if state["pending_stages"] else "COMPLETE"
    state["status"] = "COMPLETE" if state["state"] == "COMPLETE" else "RUNNABLE"
    write_state(state)


def resource_snapshot() -> dict:
    result = {"mem_available_kib": None, "gpu_rows": [], "compute_apps": [], "process_count": None}
    try:
        meminfo = Path("/proc/meminfo").read_text().splitlines()
        values = {line.split(":", 1)[0]: int(line.split()[1]) for line in meminfo if ":" in line}
        result["mem_available_kib"] = values.get("MemAvailable")
        result["process_count"] = sum(1 for _ in Path("/proc").glob("[0-9]*"))
    except Exception as exc:
        result["mem_error"] = str(exc)
    try:
        rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
        result["gpu_rows"] = [line.strip() for line in rows.splitlines() if line.strip()]
        apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader"], text=True)
        result["compute_apps"] = [line.strip() for line in apps.splitlines() if line.strip()]
    except Exception as exc:
        result["gpu_error"] = str(exc)
    return result


def run_stage(state: dict, stage: str, command: list[str], artifact: Path, *, allow_exit: tuple[int, ...] = ()) -> bool:
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode not in (0, *allow_exit):
        state["status"] = "FAILED"
        state.setdefault("failure_history", []).append({"stage": stage, "returncode": result.returncode, "time": now(), "command": command})
        write_state(state)
        return False
    if result.returncode in allow_exit:
        return False
    mark_done(state, stage, artifact)
    return True


def advance_once() -> dict:
    ensure_output_layout()
    state = read_state()
    # Bootstrap is already recorded by bootstrap.py.  The following stages
    # are deliberately CPU-only and safe while foreign GPU jobs are active.
    if state.get("state") == "BOOTSTRAP":
        mark_done(state, "BOOTSTRAP", OUTPUT_TARGET / "audit/bootstrap.json")
    if state.get("state") == "DATA_AUDIT":
        if not run_stage(state, "DATA_AUDIT", [PYTHON, str(ROOT / "scripts/trackocd_v2/audit_canonical_tao.py")], OUTPUT_TARGET / "audit/canonical_tao_universe.json"):
            return state
    if state.get("state") == "PROTOCOL_BUILD":
        if not run_stage(state, "PROTOCOL_BUILD", [PYTHON, str(ROOT / "scripts/trackocd_v2/build_gt_track_stream.py")], OUTPUT_TARGET / "audit/gt_track_stream.json"):
            return state
    if state.get("state") == "GT_FEATURE_BUILD":
        snapshot = resource_snapshot()
        artifact = OUTPUT_TARGET / "audit/common_feature_audit.json"
        build_command = [FEATURE_PYTHON, str(ROOT / "scripts/trackocd_v2/build_common_features.py"), "--split", "both", "--workers", "1"]
        result = subprocess.run(build_command, cwd=ROOT)
        if result.returncode == 0:
            audit_command = [PYTHON, str(ROOT / "scripts/trackocd_v2/audit_common_features.py")]
            audit_result = subprocess.run(audit_command, cwd=ROOT)
            if audit_result.returncode == 0:
                mark_done(state, "GT_FEATURE_BUILD", artifact)
            else:
                state["status"] = "FAILED"
                state.setdefault("failure_history", []).append({"stage": "GT_FEATURE_BUILD", "returncode": audit_result.returncode, "time": now(), "command": audit_command})
                write_state(state)
        elif result.returncode == 2:
            state["status"] = "WAITING_RESOURCE"
            state["resource_events"].append({"stage": "GT_FEATURE_BUILD", "reason": "v2 common feature build is waiting for an idle GPU; historical cache lacks required prefix16", "snapshot": snapshot, "feature_builder": build_command, "time": now()})
            write_state(state)
        else:
            state["status"] = "FAILED"
            state.setdefault("failure_history", []).append({"stage": "GT_FEATURE_BUILD", "returncode": result.returncode, "time": now(), "command": build_command})
            write_state(state)
        return state
    if state.get("state") == "GT_GEOMETRY_AUDIT":
        artifact = OUTPUT_TARGET / "audit/geometry_audit.json"
        command = [PYTHON, str(ROOT / "scripts/trackocd_v2/audit_geometry.py")]
        if not run_stage(state, "GT_GEOMETRY_AUDIT", command, artifact):
            return state
        return state
    if state.get("state") in ("GT_NEAREST", "GT_DPMEANS"):
        stage = state["state"]
        method = "nearest" if stage == "GT_NEAREST" else "dpmeans"
        artifact = OUTPUT_TARGET / ("tables/gt_%s.json" % method)
        command = [PYTHON, str(ROOT / "scripts/trackocd_v2/run_gt_baselines.py"), "--method", method]
        if not run_stage(state, stage, command, artifact):
            return state
        return state
    if state.get("state") == "GT_PHE":
        artifact = OUTPUT_TARGET / "tables/gt_phe.json"
        command = [FEATURE_PYTHON, str(ROOT / "scripts/trackocd_v2/run_gt_phe.py")]
        if not run_stage(state, "GT_PHE", command, artifact):
            return state
        return state
    if state.get("state") == "GT_CURRENT_MODEL":
        artifact = OUTPUT_TARGET / "audit/current_model_contract.json"
        command = [FEATURE_PYTHON, str(ROOT / "scripts/trackocd_v2/audit_current_model_contract.py")]
        if not run_stage(state, "GT_CURRENT_MODEL", command, artifact):
            return state
        return state
    if state.get("state") == "GT_BENCHMARK_TABLE":
        artifact = OUTPUT_TARGET / "tables/gt_benchmark_table.json"
        command = [PYTHON, str(ROOT / "scripts/trackocd_v2/build_gt_benchmark_table.py")]
        if not run_stage(state, "GT_BENCHMARK_TABLE", command, artifact):
            return state
        return state
    if state.get("state") == "PREDICTED_STREAM_BUILD":
        stream_audit = OUTPUT_TARGET / "audit/predicted_track_stream.json"
        stream_manifest = OUTPUT_TARGET / "manifests/tao_val_predicted_tracks.jsonl"
        if not (stream_audit.exists() and stream_manifest.exists() and json.loads(stream_audit.read_text()).get("status") == "COMPLETE"):
            stream_command = [PYTHON, str(ROOT / "scripts/trackocd_v2/build_predicted_stream.py")]
            stream_result = subprocess.run(stream_command, cwd=ROOT)
            if stream_result.returncode != 0:
                state["status"] = "FAILED"
                state.setdefault("failure_history", []).append({"stage": "PREDICTED_STREAM_BUILD", "returncode": stream_result.returncode, "time": now(), "command": stream_command})
                write_state(state)
                return state
        snapshot = resource_snapshot()
        feature_command = [FEATURE_PYTHON, str(ROOT / "scripts/trackocd_v2/build_common_features.py"), "--split", "pred", "--workers", "1"]
        feature_result = subprocess.run(feature_command, cwd=ROOT)
        if feature_result.returncode == 0:
            audit_command = [PYTHON, str(ROOT / "scripts/trackocd_v2/audit_predicted_features.py")]
            audit_result = subprocess.run(audit_command, cwd=ROOT)
            if audit_result.returncode == 0:
                mark_done(state, "PREDICTED_STREAM_BUILD", OUTPUT_TARGET / "audit/predicted_feature_audit.json")
            else:
                state["status"] = "FAILED"
                state.setdefault("failure_history", []).append({"stage": "PREDICTED_STREAM_BUILD", "returncode": audit_result.returncode, "time": now(), "command": audit_command})
                write_state(state)
        elif feature_result.returncode == 2:
            state["status"] = "WAITING_RESOURCE"
            state["resource_events"].append({"stage": "PREDICTED_STREAM_BUILD", "reason": "predicted feature build is waiting for an idle GPU", "snapshot": snapshot, "feature_builder": feature_command, "time": now()})
            write_state(state)
        else:
            state["status"] = "FAILED"
            state.setdefault("failure_history", []).append({"stage": "PREDICTED_STREAM_BUILD", "returncode": feature_result.returncode, "time": now(), "command": feature_command})
            write_state(state)
        return state
    if state.get("state") in ("PRED_NEAREST", "PRED_DPMEANS", "PRED_PHE"):
        stage = state["state"]
        method = {"PRED_NEAREST": "nearest", "PRED_DPMEANS": "dpmeans", "PRED_PHE": "phe"}[stage]
        artifact = OUTPUT_TARGET / ("tables/pred_%s.json" % method)
        command = [FEATURE_PYTHON, str(ROOT / "scripts/trackocd_v2/run_pred_baselines.py"), "--method", method]
        if not run_stage(state, stage, command, artifact):
            return state
        return state
    if state.get("state") == "PRED_CURRENT_MODEL":
        artifact = OUTPUT_TARGET / "audit/predicted_current_model_contract.json"
        command = [PYTHON, str(ROOT / "scripts/trackocd_v2/audit_predicted_current_model.py")]
        if not run_stage(state, "PRED_CURRENT_MODEL", command, artifact):
            return state
        return state
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=1200)
    args = parser.parse_args()
    if not args.loop:
        print(json.dumps(advance_once(), indent=2, sort_keys=True))
        return
    while True:
        state = advance_once()
        if state.get("state") == "COMPLETE":
            return
        time.sleep(max(60, args.interval_seconds))


if __name__ == "__main__":
    main()
