#!/usr/bin/env python3
"""Record whether the physical-frontend bake-off may safely start.

This is a read-only preflight.  It does not run a frontend, open images, read
Test semantic labels, or terminate any process.  The result distinguishes a
temporary resource wait from an incomplete same-protocol bake-off so a later
resume can make the smallest justified next move.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.trackocd_v2.frontend_contract import evaluator_contract, route_specs  # noqa: E402
from src.trackocd_v2.io import OUTPUT_TARGET, atomic_json, ensure_output_layout  # noqa: E402


OUTPUT = OUTPUT_TARGET / "audit/frontend_execution_preflight.json"
ROUTE_PLAN = OUTPUT_TARGET / "audit/frontend_bakeoff_route.json"
SELECTION = OUTPUT_TARGET / "audit/frontend_selection.json"
REPRESENTATION = OUTPUT_TARGET / "audit/representation_decision.json"
STAGE_PATHS = {
    "SimOWT/Q0": OUTPUT_TARGET / "audit/frontend_simowt.json",
    "OVTR-native": OUTPUT_TARGET / "audit/frontend_ovtr.json",
    "COVTrack-native": OUTPUT_TARGET / "audit/frontend_covtrack_native.json",
    "COVTrack-NoSemantic": OUTPUT_TARGET / "audit/frontend_covtrack_nosem.json",
}
PYTHON = Path(os.environ.get("TRACKOCD_FEATURE_PYTHON", "/home/lwr/anaconda3/envs/aglldiff/bin/python"))


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _read_meminfo() -> dict[str, int | None]:
    values: dict[str, int | None] = {"MemTotal": None, "MemAvailable": None, "SwapTotal": None, "SwapFree": None}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, _, rest = line.partition(":")
            if key in values:
                values[key] = int(rest.split()[0])
    except (OSError, ValueError):
        pass
    return values


def _proc_cmdline(pid: int) -> str | None:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return None
    return raw.replace(b"\0", b" ").decode(errors="replace").strip()


def _task_owned(cmdline: str | None) -> bool:
    if not cmdline:
        return False
    needles = (str(ROOT), str(OUTPUT_TARGET))
    return any(needle in cmdline for needle in needles)


def _resource_snapshot() -> dict[str, Any]:
    mem = _read_meminfo()
    total = mem["MemTotal"]
    available = mem["MemAvailable"]
    safety_floor = int(total * 0.25) if total is not None else None
    snapshot: dict[str, Any] = {
        "mem_total_kib": total,
        "mem_available_kib": available,
        "mem_safety_floor_kib": safety_floor,
        "ram_floor_pass": bool(available is not None and safety_floor is not None and available >= safety_floor),
        "swap_total_kib": mem["SwapTotal"],
        "swap_free_kib": mem["SwapFree"],
        "process_count": len(list(Path("/proc").glob("[0-9]*"))),
        "gpu_rows": [],
        "compute_apps": [],
    }
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        snapshot["gpu_query_returncode"] = gpu.returncode
        snapshot["gpu_rows"] = [line.strip() for line in gpu.stdout.splitlines() if line.strip()]
        if gpu.stderr:
            snapshot["gpu_query_stderr"] = gpu.stderr[-2000:]
    except (OSError, subprocess.TimeoutExpired) as exc:
        snapshot["gpu_query_error"] = str(exc)
    try:
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        for line in apps.stdout.splitlines():
            if not line.strip():
                continue
            fields = [part.strip() for part in line.split(",")]
            pid: int | None
            try:
                pid = int(fields[1])
            except (IndexError, ValueError):
                pid = None
            cmdline = _proc_cmdline(pid) if pid is not None else None
            snapshot["compute_apps"].append({
                "raw": line.strip(),
                "gpu_uuid": fields[0] if fields else None,
                "pid": pid,
                "cmdline": cmdline,
                "task_owned": _task_owned(cmdline),
            })
        snapshot["compute_query_returncode"] = apps.returncode
        if apps.stderr:
            snapshot["compute_query_stderr"] = apps.stderr[-2000:]
    except (OSError, subprocess.TimeoutExpired) as exc:
        snapshot["compute_query_error"] = str(exc)
    occupied_gpu_uuids = {app["gpu_uuid"] for app in snapshot["compute_apps"] if app.get("gpu_uuid")}
    snapshot["occupied_gpu_count"] = len(occupied_gpu_uuids)
    snapshot["idle_gpu_count"] = max(0, len(snapshot["gpu_rows"]) - len(occupied_gpu_uuids))
    snapshot["foreign_compute_app_count"] = sum(1 for app in snapshot["compute_apps"] if not app["task_owned"])
    snapshot["trackocd_compute_app_count"] = sum(1 for app in snapshot["compute_apps"] if app["task_owned"])
    return snapshot


def _dependency_probe() -> dict[str, Any]:
    """Inspect import availability without importing the heavy frameworks."""

    # Dotted ``find_spec`` calls import their parent package.  That is too
    # expensive and can allocate framework state during a read-only preflight;
    # top-level spec lookup is sufficient to flag the known environment split.
    modules = ["torch", "mmcv", "mmdet", "clip", "diffusers", "tao"]
    result: dict[str, Any] = {"python": str(PYTHON), "exists": PYTHON.is_file(), "modules": {}}
    if not PYTHON.is_file():
        return result
    code = (
        "import importlib.util, json; names = %r; "
        "print(json.dumps({n: bool(importlib.util.find_spec(n)) for n in names}, sort_keys=True))"
    ) % modules
    try:
        proc = subprocess.run([str(PYTHON), "-c", code], capture_output=True, text=True, timeout=30, check=False)
        result["returncode"] = proc.returncode
        if proc.stdout.strip():
            result["modules"] = json.loads(proc.stdout.strip().splitlines()[-1])
        if proc.stderr:
            result["stderr_tail"] = proc.stderr[-2000:]
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        result["error"] = str(exc)
    return result


def _stage_assets() -> dict[str, Any]:
    result: dict[str, Any] = {}
    routes = route_specs(ROOT)
    for name, path in STAGE_PATHS.items():
        item: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                item.update({
                    "status": payload.get("status"),
                    "native_run_complete": payload.get("native_run_complete") is True,
                    "same_v2_predicted_stream": payload.get("same_v2_predicted_stream") is True,
                    "same_v2_physical_metric_protocol": payload.get("same_v2_physical_metric_protocol") is True,
                    "same_v2_evaluator_contract": payload.get("same_v2_evaluator_contract") is True,
                    "physical_stream_contract_complete": payload.get("physical_stream_contract_complete") is True,
                    "requested_metrics_complete": payload.get("requested_metrics_complete") is True,
                    "historical_reference_only": payload.get("historical_reference_only"),
                    "frontend_role": payload.get("frontend_role", routes.get(name, {}).get("frontend_role")),
                    "candidate_for_final_frontend": payload.get(
                        "candidate_for_final_frontend",
                        routes.get(name, {}).get("candidate_for_final_frontend"),
                    ),
                })
            except (OSError, json.JSONDecodeError) as exc:
                item["read_error"] = str(exc)
        result[name] = item
    return result


def build_payload() -> dict[str, Any]:
    resources = _resource_snapshot()
    stages = _stage_assets()
    selection_payload: dict[str, Any] = {}
    representation_payload: dict[str, Any] = {}
    if SELECTION.is_file():
        try:
            selection_payload = json.loads(SELECTION.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            selection_payload = {}
    if REPRESENTATION.is_file():
        try:
            representation_payload = json.loads(REPRESENTATION.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            representation_payload = {}
    selected_frontend = selection_payload.get("selected_frontend")
    selection_complete = selection_payload.get("status") == "FINAL_PHYSICAL_FRONTEND_SELECTED" and bool(selected_frontend)
    representation_complete = representation_payload.get("status") == "FINAL_REPRESENTATION_SELECTED"
    formal_cache_authorized = representation_complete and representation_payload.get("formal_common_feature_cache_authorized") is True
    comparable = [
        name for name, item in stages.items()
        if item.get("native_run_complete")
        and item.get("same_v2_evaluator_contract")
        and item.get("physical_stream_contract_complete")
        and item.get("same_v2_physical_metric_protocol")
        and item.get("requested_metrics_complete")
    ]
    reasons: list[str] = []
    if not resources["ram_floor_pass"]:
        reasons.append("MemAvailable is below the >=25% total-RAM safety floor")
    if not resources["idle_gpu_count"]:
        reasons.append("all visible GPUs have compute applications; no idle GPU is available")
    if "COVTrack-NoSemantic" not in comparable:
        reasons.append("COVTrack-NoSemantic has no completed native output with the shared evaluator contract")
    if not comparable:
        reasons.append("no frontend has completed an independent native physical stream plus the shared v2 metric contract")
    if reasons:
        if formal_cache_authorized and (not resources["ram_floor_pass"] or not resources["idle_gpu_count"]):
            status = "WAITING_RESOURCE_FOR_FORMAL_CACHE"
        else:
            status = "WAITING_RESOURCE_AND_FRONTEND_ASSETS" if not resources["ram_floor_pass"] or not resources["idle_gpu_count"] else "WAITING_FRONTEND_ASSETS"
    else:
        status = "READY_FOR_FORMAL_PREDICTED_CACHE" if formal_cache_authorized else "READY_FOR_FRONTEND_BAKEOFF_REVIEW"
    if formal_cache_authorized:
        formal_cache_reason = f"FINAL_PHYSICAL_FRONTEND={selected_frontend} is selected and representation decision authorizes the formal cache"
        next_action = "wait for the RAM/GPU resource gate to clear, then build the selected frontend's sharded formal predicted feature cache"
    elif selection_complete:
        formal_cache_reason = f"FINAL_PHYSICAL_FRONTEND={selected_frontend} is selected; representation decision has not yet authorized the formal cache"
        next_action = "complete the representation decision, then wait for the RAM/GPU resource gate before building the formal predicted cache"
    else:
        formal_cache_reason = "FINAL_PHYSICAL_FRONTEND is not selected"
        next_action = "wait for the independent native frontend bake-off to select a FINAL_PHYSICAL_FRONTEND before building the formal predicted cache"
    return {
        "schema_version": "trackocd.v2.frontend_execution_preflight.v1",
        "status": status,
        "generated_utc": _now(),
        "project_root": str(ROOT),
        "read_only": True,
        "native_inference_started": False,
        "images_opened": False,
        "test_semantic_accessed": False,
        "resources": resources,
        "dependencies": _dependency_probe(),
        "route_plan": {"path": str(ROUTE_PLAN.resolve()), "exists": ROUTE_PLAN.is_file()},
        "evaluator_contract": evaluator_contract(ROOT),
        "frontend_stage_assets": stages,
        "same_protocol_comparable_frontends": comparable,
        "frontend_selection": {
            "path": str(SELECTION.resolve()),
            "exists": SELECTION.is_file(),
            "status": selection_payload.get("status"),
            "selected_frontend": selected_frontend,
        },
        "representation_decision": {
            "path": str(REPRESENTATION.resolve()),
            "exists": REPRESENTATION.is_file(),
            "status": representation_payload.get("status"),
            "final_physical_frontend": representation_payload.get("final_physical_frontend"),
        },
        "formal_common_feature_cache_authorized": formal_cache_authorized,
        "formal_cache_reason": formal_cache_reason,
        "reasons": reasons,
        "next_action": next_action,
        "external_processes_touched": False,
    }


def main() -> int:
    out = ensure_output_layout()
    payload = build_payload()
    atomic_json(out / "audit/frontend_execution_preflight.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
