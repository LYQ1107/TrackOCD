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
FEATURE_PYTHON = os.environ.get("TRACKOCD_FEATURE_PYTHON", "/home/lwr/anaconda3/envs/aglldiff/bin/python")
if not Path(FEATURE_PYTHON).is_file():
    FEATURE_PYTHON = PYTHON
sys.path.insert(0, str(ROOT))

from src.trackocd_v2.io import OUTPUT_TARGET, atomic_json, ensure_output_layout  # noqa: E402
from src.trackocd_v2.protocol import STAGES, assert_test_semantic_access_allowed  # noqa: E402


STATE_PATH = OUTPUT_TARGET / "audit/autonomous_state.json"
FRONTEND_STAGE_SPECS = {
    "FRONTEND_SIMOWT": ("simowt", "frontend_simowt.json"),
    "FRONTEND_OVTR": ("ovtr", "frontend_ovtr.json"),
    "FRONTEND_COVTRACK_NATIVE": ("covtrack_native", "frontend_covtrack_native.json"),
    "FRONTEND_COVTRACK_NOSEM": ("covtrack_nosem", "frontend_covtrack_nosem.json"),
}
FRONTEND_STAGE_ORDER = tuple(FRONTEND_STAGE_SPECS)
FRONTEND_NAMES = {
    "simowt": "SimOWT/Q0",
    "ovtr": "OVTR-native",
    "covtrack_native": "COVTrack-native",
    "covtrack_nosem": "COVTrack-NoSemantic",
}
FRONTEND_SLUGS = {name: slug for slug, name in FRONTEND_NAMES.items()}
TEST_ANNOTATION = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/tao/annotations/tao_test_lvis_v1_classes.json"
)
FINAL_FREEZE = OUTPUT_TARGET / "audit/FINAL_FREEZE.json"


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


def _frontend_preflight() -> dict:
    """Run one read-only resource/dependency snapshot for a frontend stage."""

    command = [PYTHON, str(ROOT / "scripts/trackocd_v2/frontend_execution_preflight.py")]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"frontend preflight failed with return code {result.returncode}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("frontend preflight did not emit JSON") from exc


def _idle_gpu_indices(preflight: dict) -> list[int]:
    rows = preflight.get("resources", {}).get("gpu_rows", [])
    gpu_records: list[tuple[int, str | None]] = []
    for row in rows:
        try:
            fields = [field.strip() for field in str(row).split(",")]
            gpu_records.append((int(fields[0]), fields[1] if len(fields) > 1 else None))
        except (IndexError, ValueError):
            continue
    occupied = {
        str(app.get("gpu_uuid"))
        for app in preflight.get("resources", {}).get("compute_apps", [])
        if app.get("gpu_uuid")
    }
    # The preflight exposes GPU UUIDs for applications and indexed GPU rows;
    # use its aggregate idle count as the hard gate.  The formal COVTrack
    # runner independently validates the exact requested index before launch.
    if not preflight.get("resources", {}).get("idle_gpu_count"):
        return []
    if not occupied:
        return [index for index, _uuid in gpu_records]
    return [index for index, uuid in gpu_records if uuid not in occupied]


def _wait_for_frontend_resources(state: dict, stage: str, *, require_gpu: bool) -> tuple[dict, int | None] | None:
    preflight = _frontend_preflight()
    resources = preflight.get("resources", {})
    ram_ok = resources.get("ram_floor_pass") is True
    idle_indices = _idle_gpu_indices(preflight)
    if not ram_ok or (require_gpu and not idle_indices):
        state["status"] = "WAITING_RESOURCE"
        state["blocking_reason"] = (
            f"{stage} is waiting for the RAM safety floor and/or an idle GPU; "
            "no task-owned worker was launched"
        )
        state.setdefault("resource_events", []).append({
            "stage": stage,
            "reason": "RAM safety floor or required idle GPU is unavailable",
            "preflight": preflight,
            "time": now(),
        })
        state["frontend_execution_preflight"] = str((OUTPUT_TARGET / "audit/frontend_execution_preflight.json").resolve())
        write_state(state)
        return None
    return preflight, (idle_indices[0] if require_gpu and idle_indices else None)


def _frontend_incomplete_stage() -> str | None:
    for stage, (_frontend, filename) in FRONTEND_STAGE_SPECS.items():
        path = OUTPUT_TARGET / "audit" / filename
        if not path.is_file():
            return stage
        try:
            status = json.loads(path.read_text()).get("status")
        except (OSError, json.JSONDecodeError):
            return stage
        if status != "BAKEOFF_METRICS_COMPLETE":
            return stage
    return None


def _rewind_incomplete_frontend_bakeoff(state: dict) -> bool:
    """Undo the old asset-only completion records before selection resumes."""

    if state.get("state") not in {"FRONTEND_SELECTION", "REPRESENTATION_DECISION"}:
        return False
    first = _frontend_incomplete_stage()
    if first is None:
        return False
    start = FRONTEND_STAGE_ORDER.index(first)
    completed = set(state.get("completed_stages", []))
    for stage in FRONTEND_STAGE_ORDER[start:]:
        completed.discard(stage)
    completed.discard("FRONTEND_SELECTION")
    state["completed_stages"] = [stage for stage in STAGES if stage in completed]
    state["pending_stages"] = [stage for stage in STAGES if stage not in completed]
    state["state"] = first
    state["status"] = "RUNNABLE"
    state["blocking_reason"] = "frontend asset-only completion was insufficient; shared v2 metrics must run before selection"
    write_state(state)
    return True


def _record_failure(state: dict, stage: str, command: list[str], returncode: int, reason: str | None = None) -> None:
    state["status"] = "FAILED"
    item = {"stage": stage, "returncode": int(returncode), "time": now(), "command": command}
    if reason:
        item["reason"] = reason
    state.setdefault("failure_history", []).append(item)
    write_state(state)


def _register_frontend_metrics(state: dict, stage: str, frontend: str, artifact: Path) -> dict:
    command = [PYTHON, str(ROOT / "scripts/trackocd_v2/register_frontend_metrics.py"), "--frontend", frontend]
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode != 0:
        _record_failure(state, stage, command, result.returncode, "shared frontend metrics could not be registered")
        return state
    refreshed = json.loads(artifact.read_text())
    if refreshed.get("status") == "BAKEOFF_METRICS_COMPLETE":
        mark_done(state, stage, artifact)
    else:
        state["status"] = "FAILED"
        state.setdefault("failure_history", []).append({
            "stage": stage,
            "returncode": 1,
            "time": now(),
            "reason": "metrics registration returned without BAKEOFF_METRICS_COMPLETE",
        })
        write_state(state)
    return state


def _advance_frontend_metrics(state: dict, stage: str, frontend: str, artifact: Path) -> dict:
    metric_path = OUTPUT_TARGET / "audit" / f"frontend_{frontend}_metrics.json"
    if metric_path.is_file():
        metric = json.loads(metric_path.read_text())
        if metric.get("status") == "COMPLETE":
            return _register_frontend_metrics(state, stage, frontend, artifact)

    if _wait_for_frontend_resources(state, stage, require_gpu=False) is None:
        return state
    run_id = "v2_val_full_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{os.getpid()}"
    command = [
        FEATURE_PYTHON,
        str(ROOT / "scripts/trackocd_v2/run_frontend_metrics.py"),
        "--frontend", frontend,
        "--run-id", run_id,
    ]
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode != 0:
        _record_failure(state, stage, command, result.returncode, "shared frontend metrics execution failed")
        return state
    if not metric_path.is_file() or json.loads(metric_path.read_text()).get("status") != "COMPLETE":
        _record_failure(state, stage, command, 1, "metrics command returned without a complete audit")
        return state
    return _register_frontend_metrics(state, stage, frontend, artifact)


def _advance_covtrack_native(state: dict, stage: str, frontend: str) -> dict:
    native_audit = OUTPUT_TARGET / "audit" / f"frontend_{frontend}_native_run.json"
    if native_audit.is_file() and json.loads(native_audit.read_text()).get("status") == "COMPLETE":
        state["status"] = "RUNNABLE"
        write_state(state)
        return state
    gate = _wait_for_frontend_resources(state, stage, require_gpu=True)
    if gate is None:
        return state
    _preflight, gpu_index = gate
    if gpu_index is None:
        state["status"] = "WAITING_RESOURCE"
        write_state(state)
        return state
    run_id = "v2_val_native_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{os.getpid()}"
    command = [
        PYTHON,
        str(ROOT / "scripts/trackocd_v2/run_covtrack_native.py"),
        "--frontend", frontend,
        "--gpu-index", str(gpu_index),
        "--run-id", run_id,
    ]
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode == 2:
        state["status"] = "WAITING_RESOURCE"
        state.setdefault("resource_events", []).append({"stage": stage, "reason": "COVTrack wrapper reported a resource wait", "time": now()})
        write_state(state)
        return state
    if result.returncode != 0:
        _record_failure(state, stage, command, result.returncode, "COVTrack native route failed")
        return state
    state["status"] = "RUNNABLE"
    write_state(state)
    return state


def _frozen_frontend(state: dict, stage: str) -> tuple[str, str] | None:
    """Return the immutable selected frontend, or record a safe wait."""

    try:
        assert_test_semantic_access_allowed(FINAL_FREEZE, f"advance {stage}")
    except RuntimeError as exc:
        state["status"] = "WAITING_FINAL_FREEZE"
        state["blocking_reason"] = str(exc)
        write_state(state)
        return None
    freeze = json.loads(FINAL_FREEZE.read_text(encoding="utf-8"))
    name = str(freeze.get("frontend") or "")
    slug = FRONTEND_SLUGS.get(name)
    if slug is None:
        state["status"] = "FAILED"
        state.setdefault("failure_history", []).append({
            "stage": stage,
            "returncode": 1,
            "time": now(),
            "reason": f"FINAL_FREEZE has unknown frontend {name}",
        })
        write_state(state)
        return None
    return slug, name


def _test_stage_path(slug: str) -> Path:
    return OUTPUT_TARGET / "audit" / f"frontend_{slug}_test.json"


def _test_native_audit_path(slug: str) -> Path:
    return OUTPUT_TARGET / "audit" / f"frontend_{slug}_native_test_run.json"


def _register_test_stream_if_needed(state: dict, stage: str, slug: str, name: str, audit: dict) -> bool:
    native_path = Path(str(audit.get("native_output", ""))).resolve()
    if not native_path.is_file():
        _record_failure(state, stage, [], 1, "native Test audit has no readable output for registration")
        return False
    command = [
        PYTHON,
        str(ROOT / "scripts/trackocd_v2/register_test_frontend_stream.py"),
        "--frontend", name,
        "--native-input", str(native_path),
        "--input-format", "tao_json",
        "--annotation", str(TEST_ANNOTATION.resolve()),
    ]
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode != 0:
        _record_failure(state, stage, command, result.returncode, "post-inference Test stream registration failed")
        return False
    return _test_stage_path(slug).is_file() and json.loads(_test_stage_path(slug).read_text()).get("status") == "FINAL_TEST_NATIVE_STREAM_NORMALIZED"


def _advance_test_frontend(state: dict) -> dict:
    stage_name = "TAO_TEST_FRONTEND"
    selected = _frozen_frontend(state, stage_name)
    if selected is None:
        return state
    slug, name = selected
    artifact = _test_stage_path(slug)
    if artifact.is_file():
        payload = json.loads(artifact.read_text())
        if payload.get("status") == "FINAL_TEST_NATIVE_STREAM_NORMALIZED":
            mark_done(state, stage_name, artifact)
            return state
    native_audit_path = _test_native_audit_path(slug)
    if native_audit_path.is_file():
        native_audit = json.loads(native_audit_path.read_text())
        if native_audit.get("status") == "COMPLETE":
            if _register_test_stream_if_needed(state, stage_name, slug, name, native_audit):
                mark_done(state, stage_name, artifact)
            return state

    gate = _wait_for_frontend_resources(state, stage_name, require_gpu=True)
    if gate is None:
        return state
    _preflight, gpu_index = gate
    if gpu_index is None:
        state["status"] = "WAITING_RESOURCE"
        write_state(state)
        return state
    run_id = "v2_test_native_" + slug + "_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{os.getpid()}"
    if slug == "simowt":
        script = ROOT / "scripts/trackocd_v2/run_simowt_native.py"
        command = [PYTHON, str(script), "--gpu-index", str(gpu_index), "--run-id", run_id]
    elif slug == "ovtr":
        script = ROOT / "scripts/trackocd_v2/run_ovtr_native.py"
        command = [PYTHON, str(script), "--gpu-index", str(gpu_index), "--run-id", run_id]
    else:
        script = ROOT / "scripts/trackocd_v2/run_covtrack_native.py"
        command = [PYTHON, str(script), "--frontend", slug, "--gpu-index", str(gpu_index), "--run-id", run_id, "--split", "test"]
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode == 2:
        state["status"] = "WAITING_RESOURCE"
        state.setdefault("resource_events", []).append({"stage": stage_name, "reason": "native Test frontend wrapper reported a resource wait", "time": now()})
        write_state(state)
        return state
    if result.returncode != 0:
        _record_failure(state, stage_name, command, result.returncode, "selected frozen Test frontend route failed")
        return state
    if not artifact.is_file() or json.loads(artifact.read_text()).get("status") != "FINAL_TEST_NATIVE_STREAM_NORMALIZED":
        _record_failure(state, stage_name, command, 1, "native Test route returned without a finalized normalized stream")
        return state
    mark_done(state, stage_name, artifact)
    return state


def _advance_test_feature_cache(state: dict) -> dict:
    stage_name = "TAO_TEST_FEATURE_CACHE"
    selected = _frozen_frontend(state, stage_name)
    if selected is None:
        return state
    slug, name = selected
    frontend_stage = _test_stage_path(slug)
    if not frontend_stage.is_file() or json.loads(frontend_stage.read_text()).get("status") != "FINAL_TEST_NATIVE_STREAM_NORMALIZED":
        state["status"] = "WAITING_TEST_FRONTEND"
        state["blocking_reason"] = "selected frozen Test frontend stream is not normalized"
        write_state(state)
        return state
    artifact = OUTPUT_TARGET / "features/formal" / slug / "test/cache_manifest.json"
    if artifact.is_file() and json.loads(artifact.read_text()).get("status") == "COMPLETE":
        mark_done(state, stage_name, artifact)
        return state
    gate = _wait_for_frontend_resources(state, stage_name, require_gpu=True)
    if gate is None:
        return state
    _preflight, gpu_index = gate
    command = [
        FEATURE_PYTHON,
        str(ROOT / "scripts/trackocd_v2/build_sharded_feature_cache.py"),
        "--frontend", slug,
        "--split", "test",
        "--batch", "8",
        "--shard-tracks", "512",
        "--gpu-index", str(gpu_index),
    ]
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode == 2:
        state["status"] = "WAITING_RESOURCE"
        state.setdefault("resource_events", []).append({"stage": stage_name, "reason": "Test sharded feature cache reported a resource wait", "time": now()})
        write_state(state)
        return state
    if result.returncode != 0:
        _record_failure(state, stage_name, command, result.returncode, "frozen Test sharded feature cache failed")
        return state
    if not artifact.is_file() or json.loads(artifact.read_text()).get("status") != "COMPLETE":
        _record_failure(state, stage_name, command, 1, "Test cache returned without a complete manifest")
        return state
    mark_done(state, stage_name, artifact)
    return state


def _advance_test_ocd(state: dict) -> dict:
    stage_name = "TAO_TEST_STANDARD_OCD"
    selected = _frozen_frontend(state, stage_name)
    if selected is None:
        return state
    slug, _name = selected
    artifact = OUTPUT_TARGET / "tables/test_ocd.json"
    if artifact.is_file() and json.loads(artifact.read_text()).get("status") == "COMPLETE":
        mark_done(state, stage_name, artifact)
        return state
    cache = OUTPUT_TARGET / "features/formal" / slug / "test/cache_manifest.json"
    if not cache.is_file() or json.loads(cache.read_text()).get("status") != "COMPLETE":
        state["status"] = "WAITING_TEST_FEATURE_CACHE"
        state["blocking_reason"] = "Test OCD is gated on the selected frontend's complete sharded cache"
        write_state(state)
        return state
    gate = _wait_for_frontend_resources(state, stage_name, require_gpu=True)
    if gate is None:
        return state
    _preflight, gpu_index = gate
    command = [FEATURE_PYTHON, str(ROOT / "scripts/trackocd_v2/run_test_ocd.py"), "--device", "cuda:0"]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    result = subprocess.run(command, cwd=ROOT, env=env, check=False)
    if result.returncode != 0:
        _record_failure(state, stage_name, command, result.returncode, "frozen Test OCD decisions/evaluation failed")
        return state
    if not artifact.is_file() or json.loads(artifact.read_text()).get("status") != "COMPLETE":
        _record_failure(state, stage_name, command, 1, "Test OCD returned without a complete table")
        return state
    mark_done(state, stage_name, artifact)
    return state


def _advance_frontend_stage(state: dict, stage: str, frontend: str, artifact: Path) -> dict:
    if not artifact.is_file():
        command = [PYTHON, str(ROOT / "scripts/trackocd_v2/register_frontend_stage.py"), "--frontend", frontend]
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode != 0:
            _record_failure(state, stage, command, result.returncode, "frontend asset registration failed")
        else:
            state["status"] = "RUNNABLE"
            write_state(state)
        return state
    payload = json.loads(artifact.read_text())
    status = payload.get("status")
    if status == "BAKEOFF_METRICS_COMPLETE":
        mark_done(state, stage, artifact)
        return state
    if status == "NATIVE_STREAM_NORMALIZED":
        return _advance_frontend_metrics(state, stage, frontend, artifact)
    if frontend in {"covtrack_native", "covtrack_nosem"} and status in {"ASSET_REGISTERED", "NOT_YET_RUN"}:
        return _advance_covtrack_native(state, stage, frontend)
    state["status"] = "WAITING_FRONTEND_EXECUTION"
    state["blocking_reason"] = f"frontend {frontend} has stage status {status}; no valid v2 stream is registered"
    write_state(state)
    return state


def advance_once() -> dict:
    ensure_output_layout()
    state = read_state()
    if _rewind_incomplete_frontend_bakeoff(state):
        return state
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
        # Protocol reprioritization: this stage registers only the public
        # physical stream.  The old branch that launched the all-prefix DINO
        # builder is intentionally removed; representation follows frontend
        # selection and is not a prerequisite for this metadata stage.
        pause_record = OUTPUT_TARGET / "audit/predicted_feature_pause.json"
        state["predicted_feature_route"] = {
            "status": "PAUSED_BY_PROTOCOL_REPRIORITIZATION",
            "pause_record": str(pause_record.resolve()) if pause_record.exists() else None,
            "full_predicted_feature_build_auto_resume": False,
            "reason": "physical frontend selection precedes formal common-feature caching",
        }
        mark_done(state, "PREDICTED_STREAM_BUILD", stream_audit)
        return state
    if state.get("state") == "PREDICTED_STREAM_METADATA_AUDIT":
        artifact = OUTPUT_TARGET / "audit/predicted_stream_statistics.json"
        command = [PYTHON, str(ROOT / "scripts/trackocd_v2/audit_predicted_stream_statistics.py")]
        if not run_stage(state, "PREDICTED_STREAM_METADATA_AUDIT", command, artifact):
            return state
        return state
    if state.get("state") == "PREDICTED_SANITY_BENCHMARK":
        artifact = OUTPUT_TARGET / "tables/predicted_sanity_benchmark.json"
        if artifact.exists() and json.loads(artifact.read_text()).get("status", "").startswith("SANITY_COMPLETE"):
            mark_done(state, "PREDICTED_SANITY_BENCHMARK", artifact)
            return state
        command = [FEATURE_PYTHON, str(ROOT / "scripts/trackocd_v2/run_predicted_sanity.py")]
        if not run_stage(state, "PREDICTED_SANITY_BENCHMARK", command, artifact):
            return state
        return state
    if state.get("state") == "FRONTEND_AUDIT":
        artifact = OUTPUT_TARGET / "audit/frontend_asset_audit.json"
        if artifact.exists() and json.loads(artifact.read_text()).get("status") == "AUDIT_COMPLETE":
            mark_done(state, "FRONTEND_AUDIT", artifact)
            return state
        command = [PYTHON, str(ROOT / "scripts/trackocd_v2/audit_frontend_assets.py")]
        if not run_stage(state, "FRONTEND_AUDIT", command, artifact):
            return state
        return state
    if state.get("state") in FRONTEND_STAGE_SPECS:
        stage = state["state"]
        frontend, filename = FRONTEND_STAGE_SPECS[stage]
        artifact = OUTPUT_TARGET / "audit" / filename
        return _advance_frontend_stage(state, stage, frontend, artifact)
    if state.get("state") == "FRONTEND_SELECTION":
        artifact = OUTPUT_TARGET / "audit/frontend_selection.json"
        if artifact.exists() and json.loads(artifact.read_text()).get("status") == "FINAL_PHYSICAL_FRONTEND_SELECTED":
            mark_done(state, "FRONTEND_SELECTION", artifact)
            return state
        command = [PYTHON, str(ROOT / "scripts/trackocd_v2/select_frontend.py")]
        if not run_stage(state, "FRONTEND_SELECTION", command, artifact):
            return state
        return state
    if state.get("state") == "REPRESENTATION_DECISION":
        selection = OUTPUT_TARGET / "audit/frontend_selection.json"
        artifact = OUTPUT_TARGET / "audit/representation_decision.json"
        preflight = OUTPUT_TARGET / "audit/frontend_execution_preflight.json"
        route_plan = OUTPUT_TARGET / "audit/frontend_bakeoff_route.json"
        if not route_plan.exists():
            subprocess.run([PYTHON, str(ROOT / "scripts/trackocd_v2/prepare_frontend_bakeoff.py")], cwd=ROOT, check=False)
        if not preflight.exists():
            subprocess.run([PYTHON, str(ROOT / "scripts/trackocd_v2/frontend_execution_preflight.py")], cwd=ROOT, check=False)
        if not selection.exists():
            state["status"] = "WAITING_FRONTEND_SELECTION"
            write_state(state)
            return state
        selection_payload = json.loads(selection.read_text())
        if selection_payload.get("status") != "FINAL_PHYSICAL_FRONTEND_SELECTED":
            atomic_json(artifact, {
                "schema_version": "trackocd.v2.representation_decision.v1",
                "status": "WAITING_FRONTEND_EXECUTION",
                "generated_utc": now(),
                "final_physical_frontend": None,
                "formal_common_feature_cache_authorized": False,
                "frontend_selection": str(selection.resolve()),
                "frontend_selection_status": selection_payload.get("status"),
                "frontend_execution_preflight": str(preflight.resolve()) if preflight.exists() else None,
                "frontend_bakeoff_route": str(route_plan.resolve()) if route_plan.exists() else None,
                "reason": "Representation/cache decision is held until an independent native frontend bake-off selects a FINAL_PHYSICAL_FRONTEND under the shared evaluator contract.",
                "test_semantic_accessed": False,
            })
            state["formal_common_feature_cache_authorized"] = False
            state["status"] = "WAITING_FRONTEND_EXECUTION"
            state["blocking_reason"] = "independent native frontend bake-off is incomplete; formal predicted cache is intentionally not authorized"
            state["frontend_execution_preflight"] = str(preflight.resolve()) if preflight.exists() else None
            state["frontend_bakeoff_route"] = str(route_plan.resolve()) if route_plan.exists() else None
            write_state(state)
            return state
        atomic_json(artifact, {
            "schema_version": "trackocd.v2.representation_decision.v1",
            "status": "FINAL_REPRESENTATION_SELECTED",
            "generated_utc": now(),
            "final_physical_frontend": selection_payload.get("selected_frontend"),
            "formal_common_feature_cache_authorized": True,
            "frontend_selection": str(selection.resolve()),
            "frontend_selection_status": selection_payload.get("status"),
            "frontend_bakeoff_route": str(route_plan.resolve()) if route_plan.exists() else None,
            "reason": "Use the representation registered by the selected frontend; formal cache remains sharded and atomic.",
            "test_semantic_accessed": False,
        })
        state["formal_common_feature_cache_authorized"] = True
        mark_done(state, "REPRESENTATION_DECISION", artifact)
        return state
    if state.get("state") == "FORMAL_PREDICTED_FEATURE_CACHE":
        selection_path = OUTPUT_TARGET / "audit/frontend_selection.json"
        if not selection_path.is_file():
            state["status"] = "WAITING_FRONTEND_SELECTION"
            write_state(state)
            return state
        selection = json.loads(selection_path.read_text())
        selected = str(selection.get("selected_frontend") or "")
        slug_by_name = {"SimOWT/Q0": "simowt", "OVTR-native": "ovtr", "COVTrack-native": "covtrack_native", "COVTrack-NoSemantic": "covtrack_nosem"}
        frontend = slug_by_name.get(selected)
        if selection.get("status") != "FINAL_PHYSICAL_FRONTEND_SELECTED" or frontend is None:
            state["status"] = "WAITING_FRONTEND_SELECTION"
            state["blocking_reason"] = "formal predicted cache is gated on FINAL_PHYSICAL_FRONTEND"
            write_state(state)
            return state
        artifact = OUTPUT_TARGET / "features/formal" / frontend / "cache_manifest.json"
        if artifact.is_file() and json.loads(artifact.read_text()).get("status") == "COMPLETE":
            mark_done(state, "FORMAL_PREDICTED_FEATURE_CACHE", artifact)
            return state
        gate = _wait_for_frontend_resources(state, "FORMAL_PREDICTED_FEATURE_CACHE", require_gpu=True)
        if gate is None:
            return state
        command = [
            FEATURE_PYTHON,
            str(ROOT / "scripts/trackocd_v2/build_sharded_feature_cache.py"),
            "--frontend", frontend,
            "--batch", "8",
            "--shard-tracks", "512",
        ]
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode == 2:
            state["status"] = "WAITING_RESOURCE"
            state.setdefault("resource_events", []).append({"stage": "FORMAL_PREDICTED_FEATURE_CACHE", "reason": "formal sharded cache reported a resource wait", "time": now()})
            write_state(state)
            return state
        if result.returncode != 0:
            _record_failure(state, "FORMAL_PREDICTED_FEATURE_CACHE", command, result.returncode, "formal sharded cache failed")
            return state
        if not artifact.is_file() or json.loads(artifact.read_text()).get("status") != "COMPLETE":
            _record_failure(state, "FORMAL_PREDICTED_FEATURE_CACHE", command, 1, "cache command returned without a complete manifest")
            return state
        mark_done(state, "FORMAL_PREDICTED_FEATURE_CACHE", artifact)
        return state
    if state.get("state") in ("PRED_NEAREST", "PRED_DPMEANS", "PRED_PHE"):
        stage = state["state"]
        method = {"PRED_NEAREST": "nearest", "PRED_DPMEANS": "dpmeans", "PRED_PHE": "phe"}[stage]
        artifact = OUTPUT_TARGET / ("tables/pred_%s.json" % method)
        if _wait_for_frontend_resources(state, stage, require_gpu=False) is None:
            return state
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
    if state.get("state") == "OCD_BASELINE_EXTENSIONS":
        artifact = OUTPUT_TARGET / "audit/external_ocd_baseline_audit.json"
        if artifact.is_file() and json.loads(artifact.read_text()).get("status") == "COMPLETE":
            mark_done(state, "OCD_BASELINE_EXTENSIONS", artifact)
            return state
        command = [PYTHON, str(ROOT / "scripts/trackocd_v2/audit_external_ocd_baselines.py")]
        if not run_stage(state, "OCD_BASELINE_EXTENSIONS", command, artifact):
            return state
        return state
    if state.get("state") == "SEMANTIC_ADAPTER":
        artifact = OUTPUT_TARGET / "audit/semantic_adapter_selection.json"
        if artifact.is_file() and json.loads(artifact.read_text()).get("status") == "SEMANTIC_ADAPTER_SELECTED":
            mark_done(state, "SEMANTIC_ADAPTER", artifact)
            return state
        gate = _wait_for_frontend_resources(state, "SEMANTIC_ADAPTER", require_gpu=True)
        if gate is None:
            return state
        _preflight, gpu_index = gate
        if gpu_index is None:
            state["status"] = "WAITING_RESOURCE"
            write_state(state)
            return state
        command = [
            FEATURE_PYTHON,
            str(ROOT / "scripts/trackocd_v2/train_semantic_adapter.py"),
            "--device", "cuda:0",
        ]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        result = subprocess.run(command, cwd=ROOT, env=env, check=False)
        if result.returncode != 0:
            _record_failure(state, "SEMANTIC_ADAPTER", command, result.returncode, "TRAIN-only semantic adapter failed")
            return state
        if not artifact.is_file() or json.loads(artifact.read_text()).get("status") != "SEMANTIC_ADAPTER_SELECTED":
            _record_failure(state, "SEMANTIC_ADAPTER", command, 1, "semantic adapter returned without a selection artifact")
            return state
        mark_done(state, "SEMANTIC_ADAPTER", artifact)
        return state
    if state.get("state") == "CONTROLLER_DECISION":
        artifact = OUTPUT_TARGET / "audit/controller_decision.json"
        if artifact.is_file() and json.loads(artifact.read_text()).get("status") == "CONTROLLER_CANDIDATES_REGISTERED":
            mark_done(state, "CONTROLLER_DECISION", artifact)
            return state
        command = [PYTHON, str(ROOT / "scripts/trackocd_v2/register_controller_decision.py")]
        if not run_stage(state, "CONTROLLER_DECISION", command, artifact):
            return state
        return state
    if state.get("state") == "SAFE_CONTROLLER":
        gt_artifact = OUTPUT_TARGET / "tables/gt_safe_controller.json"
        pred_artifact = OUTPUT_TARGET / "tables/pred_safe_controller.json"
        if not (gt_artifact.is_file() and json.loads(gt_artifact.read_text()).get("status") == "COMPLETE"):
            gate = _wait_for_frontend_resources(state, "SAFE_CONTROLLER", require_gpu=False)
            if gate is None:
                return state
            command = [PYTHON, str(ROOT / "scripts/trackocd_v2/run_gt_safe_controller.py"), "--prefixes", "16"]
            result = subprocess.run(command, cwd=ROOT, check=False)
            if result.returncode != 0:
                _record_failure(state, "SAFE_CONTROLLER", command, result.returncode, "GT safe-controller validation failed")
                return state
            if not gt_artifact.is_file() or json.loads(gt_artifact.read_text()).get("status") != "COMPLETE":
                _record_failure(state, "SAFE_CONTROLLER", command, 1, "GT safe-controller returned without a complete table")
                return state
            return state
        if not (pred_artifact.is_file() and json.loads(pred_artifact.read_text()).get("status") == "COMPLETE"):
            gate = _wait_for_frontend_resources(state, "SAFE_CONTROLLER", require_gpu=True)
            if gate is None:
                return state
            _preflight, gpu_index = gate
            if gpu_index is None:
                state["status"] = "WAITING_RESOURCE"
                write_state(state)
                return state
            command = [
                FEATURE_PYTHON,
                str(ROOT / "scripts/trackocd_v2/run_pred_safe_controller.py"),
                "--device", "cuda:0",
            ]
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
            result = subprocess.run(command, cwd=ROOT, env=env, check=False)
            if result.returncode != 0:
                _record_failure(state, "SAFE_CONTROLLER", command, result.returncode, "formal predicted safe-controller replay failed")
                return state
            if not pred_artifact.is_file() or json.loads(pred_artifact.read_text()).get("status") != "COMPLETE":
                _record_failure(state, "SAFE_CONTROLLER", command, 1, "predicted safe-controller returned without a complete table")
                return state
        mark_done(state, "SAFE_CONTROLLER", pred_artifact)
        return state
    if state.get("state") == "VAL_FINAL_SELECTION":
        artifact = OUTPUT_TARGET / "audit/val_final_selection.json"
        if artifact.is_file() and json.loads(artifact.read_text()).get("status") in {"FINAL_VAL_SELECTION", "FINAL_VAL_SELECTION_NO_IMPROVEMENT"}:
            mark_done(state, "VAL_FINAL_SELECTION", artifact)
            return state
        command = [PYTHON, str(ROOT / "scripts/trackocd_v2/select_val_final.py")]
        if not run_stage(state, "VAL_FINAL_SELECTION", command, artifact):
            return state
        return state
    if state.get("state") == "FINAL_FREEZE":
        artifact = OUTPUT_TARGET / "audit/FINAL_FREEZE.json"
        if artifact.is_file() and json.loads(artifact.read_text()).get("status") == "FINAL_FREEZE":
            mark_done(state, "FINAL_FREEZE", artifact)
            return state
        command = [PYTHON, str(ROOT / "scripts/trackocd_v2/finalize_freeze.py")]
        if not run_stage(state, "FINAL_FREEZE", command, artifact):
            return state
        return state
    if state.get("state") == "TAO_TEST_FRONTEND":
        return _advance_test_frontend(state)
    if state.get("state") == "TAO_TEST_FEATURE_CACHE":
        return _advance_test_feature_cache(state)
    if state.get("state") == "TAO_TEST_STANDARD_OCD":
        return _advance_test_ocd(state)
    if state.get("state") == "TAO_TEST_PERSISTENT":
        artifact = OUTPUT_TARGET / "audit/test_persistent.json"
        selected = _frozen_frontend(state, "TAO_TEST_PERSISTENT")
        if selected is None:
            return state
        if artifact.is_file() and json.loads(artifact.read_text()).get("status") == "COMPLETE":
            mark_done(state, "TAO_TEST_PERSISTENT", artifact)
            return state
        command = [PYTHON, str(ROOT / "scripts/trackocd_v2/build_test_persistent_audit.py")]
        if not run_stage(state, "TAO_TEST_PERSISTENT", command, artifact):
            return state
        return state
    if state.get("state") == "TAO_TEST_TRACKING":
        selected = _frozen_frontend(state, "TAO_TEST_TRACKING")
        if selected is None:
            return state
        slug, _name = selected
        artifact = OUTPUT_TARGET / "audit" / f"frontend_{slug}_test_metrics.json"
        if artifact.is_file():
            payload = json.loads(artifact.read_text())
            if payload.get("status") == "COMPLETE" and payload.get("split") == "test":
                mark_done(state, "TAO_TEST_TRACKING", artifact)
                return state
        if not (_test_stage_path(slug).is_file() and json.loads(_test_stage_path(slug).read_text()).get("status") == "FINAL_TEST_NATIVE_STREAM_NORMALIZED"):
            state["status"] = "WAITING_TEST_FRONTEND"
            state["blocking_reason"] = "Test frontend stream is not finalized for tracking evaluation"
            write_state(state)
            return state
        gate = _wait_for_frontend_resources(state, "TAO_TEST_TRACKING", require_gpu=False)
        if gate is None:
            return state
        run_id = "v2_test_tracking_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{os.getpid()}"
        command = [
            FEATURE_PYTHON,
            str(ROOT / "scripts/trackocd_v2/run_frontend_metrics.py"),
            "--frontend", slug,
            "--split", "test",
            "--run-id", run_id,
        ]
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode != 0:
            _record_failure(state, "TAO_TEST_TRACKING", command, result.returncode, "frozen Test frontend tracking metrics failed")
            return state
        if not artifact.is_file() or json.loads(artifact.read_text()).get("status") != "COMPLETE":
            _record_failure(state, "TAO_TEST_TRACKING", command, 1, "Test tracking command returned without a complete metrics audit")
            return state
        mark_done(state, "TAO_TEST_TRACKING", artifact)
        return state
    if state.get("state") == "FINAL_TABLES":
        artifact = OUTPUT_TARGET / "tables/final_tables.json"
        if artifact.is_file() and json.loads(artifact.read_text()).get("status") == "COMPLETE":
            mark_done(state, "FINAL_TABLES", artifact)
            return state
        command = [PYTHON, str(ROOT / "scripts/trackocd_v2/build_final_tables.py")]
        if not run_stage(state, "FINAL_TABLES", command, artifact):
            return state
        if not artifact.is_file() or json.loads(artifact.read_text(encoding="utf-8")).get("status") != "COMPLETE":
            _record_failure(state, "FINAL_TABLES", command, 1, "final table command returned without a complete artifact")
            return state
        return state
    if state.get("state") == "FINAL_REPORT":
        artifact = ROOT / "docs/iclr27_phase24/PHASE24_PROPOSAL_SELECTION_SOURCE_GENERALIZATION_COMPLETE_REPORT.md"
        if artifact.is_file() and "#" in artifact.read_text(encoding="utf-8"):
            mark_done(state, "FINAL_REPORT", artifact)
            return state
        command = [PYTHON, str(ROOT / "scripts/trackocd_v2/build_final_report.py")]
        if not run_stage(state, "FINAL_REPORT", command, artifact):
            return state
        if not artifact.is_file() or "#" not in artifact.read_text(encoding="utf-8"):
            _record_failure(state, "FINAL_REPORT", command, 1, "final report command returned without a readable report")
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
